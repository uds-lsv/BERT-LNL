import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from trainers.trainer import Trainer
from tqdm import tqdm
# import wandb

from trainers.early_stopper import EarlyStopper
from trainers.loss_noise_tracker import LossNoiseTracker


class BertWN_Trainer(Trainer):
    def __init__(self, args, logger, log_dir, model_config, full_dataset, random_state):
        super(BertWN_Trainer, self).__init__(args, logger, log_dir, model_config, full_dataset, random_state)
        self.store_model_flag = True if args.store_model == 1 else False


    def train(self, args, logger, full_dataset):
        logger.info('Bert WN Trainer: training started')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        nl_set, ul_set, v_set, t_set, l2id, id2l = full_dataset
        logger.info(f'training size: {len(nl_set)}',)
        logger.info(f'validation size: {len(v_set)}' )
        logger.info(f'test size: {len(t_set)}')

        model = self.create_model(args)
        model = model.to(device)

        assert args.nl_batch_size % args.gradient_accumulation_steps == 0
        nl_sub_batch_size = args.nl_batch_size // args.gradient_accumulation_steps
        nl_bucket = torch.utils.data.DataLoader(nl_set, batch_size=nl_sub_batch_size,
                                                shuffle=True,
                                                num_workers=0)

        nl_iter = iter(nl_bucket)


        t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.eval_batch_size,
                                               shuffle=False,
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

        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(args, model)
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


        for idx in tqdm(range(num_training_steps), desc=f'[Vannilla Trainer] training'):
            ce_loss_mean = 0.0

            for i in range(args.gradient_accumulation_steps):
                model.train()
                try:
                    nl_batch = next(nl_iter)
                except:
                    nl_iter = iter(nl_bucket)
                    nl_batch = next(nl_iter)

                nll_loss = \
                    self.forward_backward_noisy_batch(model, {'nl_batch': nl_batch}, ce_loss_fn, args, device)
                ce_loss_mean += nll_loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer_scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            # wandb.log({'train/batch_loss': ce_loss_mean})

            if self.needs_eval(args, global_step):
                val_score = self.eval_model_with_both_labels(model, v_loader, device, fast_mode=args.fast_eval)
                test_score = self.eval_model(args, logger, t_loader, model, device, fast_mode=args.fast_eval)

                early_stopper.register(val_score['score_dict_n']['accuracy'], model, optimizer)

                # wandb.log({'eval/loss/val_c_loss': val_score['val_c_loss'],
                #            'eval/loss/val_n_loss': val_score['val_n_loss'],
                #            'eval/score/val_c_acc': val_score['score_dict_c']['accuracy'],
                #            'eval/score/val_n_acc': val_score['score_dict_n']['accuracy'],
                #            'eval/score/test_acc': test_score['score_dict']['accuracy']}, step=global_step)

                loss_noise_tracker.log_loss(model, global_step, device)
                loss_noise_tracker.log_last_histogram_to_wandb(step=global_step, normalize=True, tag='eval/loss')

            if early_stopper.early_stop:
                break

        if args.save_loss_tracker_information:
            loss_noise_tracker.save_logged_information()
            self.logger.info("[WN Trainer]: loss history saved")
        best_model = self.create_model(args)
        best_model_weights = early_stopper.get_final_res()["es_best_model"]
        best_model.load_state_dict(best_model_weights)
        best_model = best_model.to(device)

        val_score = self.eval_model_with_both_labels(best_model, v_loader, device, fast_mode=False)
        test_score = self.eval_model(args, logger, t_loader, best_model, device, fast_mode=False)
        # wandb.run.summary["best_score_on_val_n"] = test_score['score_dict']['accuracy']
        # wandb.run.summary["best_val_n"] = val_score['score_dict_n']['accuracy']
        # wandb.run.summary["best_val_c_on_val_n"] = val_score['score_dict_c']['accuracy']




    def forward_backward_noisy_batch(self, model, data_dict, loss_fn, args, device):

        nl_databatch = data_dict['nl_batch']
        input_ids = nl_databatch['input_ids']
        attention_mask = nl_databatch['attention_mask']
        n_labels = nl_databatch['n_labels']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        n_labels = n_labels.to(device)

        outputs = model(input_ids, attention_mask)['logits']
        loss = loss_fn(outputs, n_labels)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()


        return loss.item()


