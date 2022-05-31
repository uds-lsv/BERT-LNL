import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from tqdm import tqdm
from trainers.trainer import Trainer
# import wandb
from trainers.early_stopper import EarlyStopper
from trainers.loss_noise_tracker import LossNoiseTracker
import os


# Reimplementation of the paper: Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels
# Check https://papers.nips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html
# Note that we use the ground truth noise level, to eliminate the influence of the estimation errors

class BertCT_Trainer(Trainer):
    def __init__(self, args, logger, log_dir, model_config, full_dataset, random_state):
        super(BertCT_Trainer, self).__init__(args, logger, log_dir, model_config, full_dataset, random_state)

    def train(self, args, logger, full_dataset):
        logger.info('Bert CT Trainer: training started')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        nl_set, ul_set, v_set, t_set, l2id, id2l = full_dataset


        model1 = self.create_model(args)
        model1 = model1.to(device)
        model2 = self.create_model(args)
        model2 = model2.to(device)

        nl_bucket = torch.utils.data.DataLoader(nl_set, batch_size=args.nl_batch_size, shuffle=True, num_workers=0)

        nl_iter = iter(nl_bucket)

        if v_set is None:
            logger.info('No validation set is used here')
            v_loader = None
        else:
            logger.info('Validation set is used here')
            v_loader = torch.utils.data.DataLoader(v_set, batch_size=args.eval_batch_size,
                                                   shuffle=False,
                                                   num_workers=0)

        t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.eval_batch_size,
                                               shuffle=False,
                                               num_workers=0)

        num_training_steps = args.num_training_steps
        optimizer1, optimizer_scheduler1 = self.get_optimizer(model1, args, num_training_steps)
        optimizer2, optimizer_scheduler2 = self.get_optimizer(model2, args, num_training_steps)




        noise_level = args.noise_level
        forget_rate = noise_level * args.forget_factor
        rate_schedule = np.ones(num_training_steps) * forget_rate
        rate_schedule[:args.T_k] = np.linspace(0, forget_rate ** args.c, args.T_k)
        logger.info(f"Total Steps: {num_training_steps} ,T_k: {args.T_k}")


        if self.store_model_flag:
            early_stopper_save_dir = os.path.join(self.log_dir, 'early_stopper_model')
            if not os.path.exists(early_stopper_save_dir):
                os.makedirs(early_stopper_save_dir)
        else:
            early_stopper_save_dir = None

        early_stopper = EarlyStopper(patience=args.patience, delta=0, save_dir=early_stopper_save_dir,
                                     large_is_better=True, verbose=False, trace_func=logger.info)

        noise_tracker_dir = os.path.join(self.log_dir, 'loss_noise_tracker')
        loss_noise_tracker = LossNoiseTracker(args, logger, nl_set, noise_tracker_dir)


        global_step = 0

        # train the network
        for idx in tqdm(range(num_training_steps), desc=f'training'):
            try:
                nl_batch = next(nl_iter)
            except:
                nl_iter = iter(nl_bucket)
                nl_batch = next(nl_iter)

            loss1, loss2 = \
                self.forward_path_for_sorting_loss(model1, model2, nl_batch,
                                                   args, device)

            ce_loss1, ce_loss2, purity1, purity2 = \
                self.do_coteaching(nl_batch, (model1, model2),
                                   (optimizer1, optimizer2),
                                   (optimizer_scheduler1, optimizer_scheduler2),
                                   (loss1, loss2),
                                   rate_schedule[global_step],
                                   args, device)
            global_step += 1

            #
            # wandb.log({'train/batch_loss1': ce_loss1})


            if self.needs_eval(args, global_step):
                val_score = self.eval_model_with_both_labels(model1, v_loader, device, fast_mode=args.fast_eval)
                test_score = self.eval_model(args, logger, t_loader, model1, device, fast_mode=args.fast_eval)

                early_stopper.register(val_score['score_dict_n']['accuracy'], model1, optimizer1)

                # wandb.log({'eval/loss/val_c_loss': val_score['val_c_loss'],
                #            'eval/loss/val_n_loss': val_score['val_n_loss'],
                #            'eval/score/val_c_acc': val_score['score_dict_c']['accuracy'],
                #            'eval/score/val_n_acc': val_score['score_dict_n']['accuracy'],
                #            'eval/score/test_acc': test_score['score_dict']['accuracy']}, step=global_step)

                loss_noise_tracker.log_loss(model1, global_step, device)
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



    def train_batch(self, model, data_batch, optimizer, optimizer_scheduler, args, device):

        total_loss = 0

        input_ids_batch = data_batch['input_ids']
        attention_mask_batch = data_batch['attention_mask']
        n_labels_batch = data_batch['n_labels']

        num_samples_in_batch = len(input_ids_batch)
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        n_labels_batch = n_labels_batch.to(device)

        num_batches = int(np.ceil(num_samples_in_batch / args.nl_batch_size))

        model.zero_grad()
        for i in range(num_batches):
            start = i * args.nl_batch_size
            end = start + args.nl_batch_size
            input_ids = input_ids_batch[start:end]
            attention_mask = attention_mask_batch[start:end]
            n_labels = n_labels_batch[start:end]

            outputs = model(input_ids, attention_mask)['logits']
            loss = F.cross_entropy(outputs, n_labels, reduction='sum')
            loss = loss / num_samples_in_batch
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer_scheduler.step()  # Update learning rate schedule

        return total_loss

    def do_coteaching(self, data_batch, models, optimizers, optimizer_schedulers,
                      losses, forget_rate, args, device):

        model1, model2 = models
        optimizer1, optimizer2 = optimizers
        loss1, loss2 = losses
        optimizer_scheduler1, optimizer_scheduler2 = optimizer_schedulers

        remember_rate = 1 - forget_rate

        filtered_data1, purity1 = self.filter_data(data_batch, loss1, remember_rate, args)
        filtered_data2, purity2 = self.filter_data(data_batch, loss2, remember_rate, args)

        loss1 = self.train_batch(model1, filtered_data2, optimizer1, optimizer_scheduler1, args, device)
        loss2 = self.train_batch(model2, filtered_data1, optimizer2, optimizer_scheduler2, args, device)

        return loss1, loss2, purity1, purity2

    def filter_data(self, data_batch, loss, remember_rate, args):

        input_ids = data_batch['input_ids']
        attention_mask = data_batch['attention_mask']
        n_labels = data_batch['n_labels']

        _, sort_idx = torch.sort(loss)
        sort_idx = sort_idx[0:int(len(sort_idx) * remember_rate)]

        purity_selected = data_batch['purity'][sort_idx]
        purity = torch.sum(purity_selected).true_divide(len(purity_selected))

        return {'input_ids': input_ids[sort_idx], 'attention_mask': attention_mask[sort_idx],
                'n_labels': n_labels[sort_idx]}, purity

    def forward_path_for_sorting_loss(self, model1, model2, data_batch, args, device):

        model1.eval()
        model2.eval()

        input_ids = data_batch['input_ids']
        attention_mask = data_batch['attention_mask']
        n_labels = data_batch['n_labels']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        n_labels = n_labels.to(device)

        with torch.no_grad():
            output1 = model1(input_ids, attention_mask)['logits']
            loss1 = F.cross_entropy(output1, n_labels, reduction='none')
            output2 = model2(input_ids, attention_mask)['logits']
            loss2 = F.cross_entropy(output2, n_labels, reduction='none')

        model1.train()
        model2.train()

        return loss1.detach().cpu(), loss2.detach().cpu()

    def get_optimizer(self, model, args, num_training_steps):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        optimizer_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                              num_training_steps=num_training_steps)
        return optimizer, optimizer_scheduler