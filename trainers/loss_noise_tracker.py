import numpy as np
import torch
import pickle
import copy
import os
import wandb


class LossNoiseTracker:
    """Track the losses during training and check whether samples with lower losses are cleaner"""

    def __init__(self, args, logger, d_set, save_dir, **kwargs):
        self.args = args
        self.logger = logger
        self.d_set = d_set
        self.purity_list = np.array(self.d_set.purity_list.detach().numpy(), dtype=bool)
        self.d_loader = torch.utils.data.DataLoader(d_set, batch_size=args.eval_batch_size,
                                                shuffle=False,
                                                num_workers=0)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.save_path = os.path.join(self.save_dir, 'noise_distribution.pickle')
        self.steps = []
        self.loss_on_clean_samples_hist=[]
        self.loss_on_wrong_samples_hist=[]

    def skip_function(self):
        if self.args.noise_level == 0.0:  # data is clean, no need to track the noise
            return True
        else:
            return False

    def normalize_data(self, data):
        min_value = np.min(data)
        max_value = np.max(data)

        return (data - min_value) / (max_value - min_value)

    def log_loss(self, model, global_step, device):
        if self.skip_function():
            return
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss_list = []
        self.steps.append(global_step)

        with torch.no_grad():
            for idx, d_batch in enumerate(self.d_loader):

                input_ids = d_batch['input_ids'].to(device)
                attention_mask = d_batch['attention_mask'].to(device)
                n_labels = d_batch['n_labels'].to(device)

                y_pred = model(input_ids, attention_mask)['logits']

                loss_batch = loss_fn(y_pred, n_labels).detach().cpu().numpy()
                loss_list.append(loss_batch)

        loss_list = np.concatenate(loss_list)
        loss_on_clean_samples = loss_list[self.purity_list]
        loss_on_wrong_samples = loss_list[~self.purity_list]

        self.loss_on_clean_samples_hist.append(loss_on_clean_samples)
        self.loss_on_wrong_samples_hist.append(loss_on_wrong_samples)

    def save_logged_information(self):
        if self.skip_function():
            return

        save_dict = {'steps': self.steps, 'loss_on_clean': self.loss_on_clean_samples_hist, 'loss_on_wrong': self.loss_on_wrong_samples_hist}
        with open(self.save_path, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def log_last_histogram_to_wandb(self, step, normalize, tag):
        if self.skip_function():
            return

        loss_clean = self.loss_on_clean_samples_hist[-1]
        loss_wrong = self.loss_on_wrong_samples_hist[-1]

        if normalize:
            loss_clean = self.normalize_data(loss_clean)
            loss_wrong = self.normalize_data(loss_wrong)

        wandb.log({f"{tag}/loss_clean" : wandb.Histogram(loss_clean),
                   f"{tag}/loss_wrong": wandb.Histogram(loss_wrong)}, step=step)



