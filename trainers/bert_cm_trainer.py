import os
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from models.cm import CM
from transformers import AdamW, get_linear_schedule_with_warmup
from trainers.trainer import Trainer
from tqdm import tqdm
from trainers.early_stopper import EarlyStopper
from trainers.loss_noise_tracker import LossNoiseTracker

# Reimplementation of the paper: An Effective Label Noise Model for DNN Text Classification
# Check https://aclanthology.org/N19-1328/
# Note that we use BERT instead of textCNN

class BertCM_Trainer(Trainer):
    def __init__(self, args, logger, log_dir, model_config, full_dataset, random_state):
        super(BertCM_Trainer, self).__init__(args, logger, log_dir, model_config, full_dataset, random_state)
        self.mse_weight = args.cm_mse_weight


    def train(self, args, logger, full_dataset):
        logger.info('Bert CM Trainer: training started...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        nl_set, ul_set, v_set, t_set, l2id, id2l = full_dataset
        base_model = self.create_model(args)
        cm_model = CM(self.model_config, args, base_model)
        cm_model = cm_model.to(device)

        assert args.nl_batch_size % args.gradient_accumulation_steps == 0
        nl_sub_batch_size = args.nl_batch_size // args.gradient_accumulation_steps

        nl_bucket = torch.utils.data.DataLoader(nl_set, batch_size=nl_sub_batch_size,
                                                shuffle=True,
                                                num_workers=0)
        nl_iter = iter(nl_bucket)

        t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.eval_batch_size,
                                               shuffle=True,
                                               num_workers=0)

        if v_set is None:
            logger.info('No validation set is used here')
            v_loader = None
        else:
            logger.info('Validation set is used here')
            v_loader = torch.utils.data.DataLoader(v_set, batch_size=args.eval_batch_size,
                                                   shuffle=False,
                                                   num_workers=0)

        num_training_steps = args.num_training_steps

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in cm_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in cm_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        optimizer_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                              num_training_steps=num_training_steps)


        ce_loss_fn = nn.CrossEntropyLoss()

        if self.store_model_flag:
            early_stopper_save_dir = os.path.join(self.log_dir, 'early_stopper_model')
            if not os.path.exists(early_stopper_save_dir):
                os.makedirs(early_stopper_save_dir)
        else:
            early_stopper_save_dir = None

        # We log the validation accuracy, so, large_is_better should set to True
        early_stopper = EarlyStopper(patience=args.patience, delta=0, save_dir=early_stopper_save_dir,
                                     large_is_better=True, verbose=False, trace_func=logger.info)

        noise_tracker_dir = os.path.join(self.log_dir, 'loss_noise_tracker')
        loss_noise_tracker = LossNoiseTracker(args, logger, nl_set, noise_tracker_dir)

        global_step = 0

        # train the network
        for idx in tqdm(range(num_training_steps), desc=f'[CM Trainer] training..'):
            ce_loss_mean, l2_loss_mean, ul_ce_loss_mean = 0, 0, 0

            for i in range(args.gradient_accumulation_steps):
                cm_model.train()
                try:
                    nl_batch = next(nl_iter)
                except:
                    nl_iter = iter(nl_bucket)
                    nl_batch = next(nl_iter)

                ce_loss, l2_loss = \
                    self.forward_backward_cm_batch(cm_model, nl_batch, ce_loss_fn, args, device)
                ce_loss_mean += ce_loss
                l2_loss_mean += l2_loss


            torch.nn.utils.clip_grad_norm_(cm_model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer_scheduler.step()  # Update learning rate schedule
            cm_model.zero_grad()
            global_step += 1

            # wandb.log({'train/batch_loss': ce_loss_mean})

            if self.needs_eval(args, global_step):
                val_score = self.eval_model_with_both_labels(base_model, v_loader, device, fast_mode=args.fast_eval)
                test_score = self.eval_model(args, logger, t_loader, base_model, device, fast_mode=args.fast_eval)

                early_stopper.register(val_score['score_dict_n']['accuracy'], base_model, optimizer)

                # wandb.log({'eval/loss/val_c_loss': val_score['val_c_loss'],
                #            'eval/loss/val_n_loss': val_score['val_n_loss'],
                #            'eval/score/val_c_acc': val_score['score_dict_c']['accuracy'],
                #            'eval/score/val_n_acc': val_score['score_dict_n']['accuracy'],
                #            'eval/score/test_acc': test_score['score_dict']['accuracy']}, step=global_step)

                loss_noise_tracker.log_loss(base_model, global_step, device)
                loss_noise_tracker.log_last_histogram_to_wandb(step=global_step, normalize=True, tag='eval/loss')

            if early_stopper.early_stop:
                break

        if args.save_loss_tracker_information:
            loss_noise_tracker.save_logged_information()
            self.logger.info("[Vanilla Trainer]: loss history saved")
        best_model = self.create_model(args)
        best_model_weights = early_stopper.get_final_res()["es_best_model"]
        best_model.load_state_dict(best_model_weights)
        best_model = best_model.to(device)

        val_score = self.eval_model_with_both_labels(best_model, v_loader, device, fast_mode=False)
        test_score = self.eval_model(args, logger, t_loader, best_model, device, fast_mode=False)
        # wandb.run.summary["best_score_on_val_n"] = test_score['score_dict']['accuracy']
        # wandb.run.summary["best_val_n"] = val_score['score_dict_n']['accuracy']
        # wandb.run.summary["best_val_c_on_val_n"] = val_score['score_dict_c']['accuracy']


    def forward_backward_cm_batch(self, cm_model, nl_databatch, loss_fn, args, device):
        # https://huggingface.co/transformers/custom_datasets.html

        input_ids = nl_databatch['input_ids']
        attention_mask = nl_databatch['attention_mask']
        n_labels = nl_databatch['n_labels']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        n_labels = n_labels.to(device)

        outputs = cm_model(input_ids, attention_mask)['logits']
        ce_loss = loss_fn(outputs, n_labels)
        l2_loss = self.mse_weight * torch.sum(cm_model.noise_model ** 2)
        total_loss = (ce_loss + l2_loss)/args.gradient_accumulation_steps
        total_loss.backward()
        return ce_loss.item(), l2_loss.item()
