import torch
import os
import numpy as np
from torch.utils.data import Dataset, RandomSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as pr_score
from transformers import AutoModel
from collections import OrderedDict
from models.text_bert import TextBert

class Trainer:
    def __init__(self, args, logger, log_dir, model_config, full_dataset, random_state):
        self.args = args
        self.logger = logger
        self.model_config = model_config
        self.num_classes = self.model_config['num_classes']
        self.log_dir = log_dir
        self.label_list = None
        self.l2id = None
        self.id2l = None
        self.r_state = random_state
        self.store_model_flag = True if args.store_model == 1 else False

    def create_model(self, args):
            model = TextBert(self.model_config, None, args)
            return model

    def needs_eval(self, args, global_step):
        if global_step % args.eval_freq == 0 and global_step != 0:
            return True
        else:
            return False

    def remove_prefix(self, text, prefix):
        return text[len(prefix):] if text.startswith(prefix) else text


    def get_data_loader(self, args, input_dataset, bz, drop_last):
        sampler = RandomSampler(input_dataset)
        bbs = BucketBatchSampler(sampler, batch_size=bz, drop_last=drop_last,
                                 sort_key=lambda x: input_dataset.length_list[x], descending=True)
        bucket_loader = torch.utils.data.DataLoader(dataset=input_dataset, batch_sampler=bbs, collate_fn=cl_pad_collate)
        return bucket_loader

    def get_confusion_mat(self, n_set):
        n_labels = n_set.targets
        gt_labels = n_set.c_targets
        label_str = [self.id2l[k] for k in range(self.num_classes)]

        return confusion_matrix(gt_labels, n_labels, labels=range(self.num_classes))


    def save_model(self, logger, model, model_name):
        output_path = os.path.join(self.log_dir, model_name)
        torch.save(model.state_dict(), output_path)
        logger.info(f"model saved at: {output_path}")



    # def prepare_noisy_dataloader(self, X_noisy, y_noisy, y_noisy_c, size):
    #     x_sub_n, y_sub_n, y_sub_nc = utils.get_random_samples(size, X_noisy, y_noisy, y_noisy_c)
    #     pure_flags = self.get_pure_flags(y_sub_n, y_sub_nc)
    #     dataset = CIFAR10_Dataset(x_sub_n, y_sub_n, pure_flags)
    #     dataloader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    #     return dataloader



    def train(self, args, logger, full_dataset):
        raise NotImplementedError()


    def eval(self, model, t_loader, performance_tracker, logger, epoch, device, args):

        all_preds = []
        all_y = []
        model.eval()

        # on the test data
        for idx, batch in enumerate(t_loader):
            xs, xs_length = batch['text'], batch['length']
            xs = xs.to(device)

            y_pred = model(xs, xs_length)
            predicted = torch.max(y_pred.cpu().data, 1)[1]
            all_preds.extend(predicted.numpy())
            all_y.extend(list(batch['label']))

        classification_score = classification_report(all_y, np.array(all_preds).flatten(), self.label_list)

        return classification_score


    def log_score(self, args, logger, score_record, score_str, sw_obj, global_step, prefix, verbose=True):
        if args.metric == "accuracy":
            score = score_record['accuracy']
            # if verbose:
            #     logger.info(f"{prefix} Acc: {score}")
            sw_obj.add_scalar(f"{prefix} Acc", score, global_step)
        elif args.metric == 'f1_macro':
            score = score_record['macro avg']['f1-score']
            # if verbose:
            #     logger.info(f"{prefix} Macro F1: {score}")
            sw_obj.add_scalar(f"{prefix} Macro F1", score, global_step)
        else:
            raise NotImplementedError(f'Given metric {args.metric} not supported')

        if verbose:
            logger.info(score_str)



    def get_sub_acc(self, all_y, all_preds, n_chunks):
        chunk_size = int(len(all_y)/n_chunks)
        ys = (all_y[i:i + chunk_size] for i in range(0, len(all_y), chunk_size))
        preds = (all_preds[i:i + chunk_size] for i in range(0, len(all_preds), chunk_size))
        return [accuracy_score(y, p) for y, p in zip(ys, preds)]

    def eval_model(self, args, logger, t_loader, model, device, fast_mode):
        all_preds = []
        all_y = []
        model.eval()

        if fast_mode:
            n_batch = len(t_loader)/10

        with torch.no_grad():
            for idx, t_batch in enumerate(t_loader):
                input_ids = t_batch['input_ids'].to(device)
                attention_mask = t_batch['attention_mask'].to(device)
                c_targets = t_batch['c_labels'].to(device)

                y_pred = model(input_ids, attention_mask)['logits']
                predicted = torch.max(y_pred.cpu(), 1)[1]
                all_preds.extend(predicted.numpy())
                all_y.extend(list(c_targets.cpu()))
                if fast_mode and idx > n_batch:
                    break

            classification_score_dict = classification_report(all_y, np.array(all_preds).flatten(),
                                                              target_names=self.label_list, output_dict=True)
            classification_score_str = classification_report(all_y, np.array(all_preds).flatten(),
                                                             target_names=self.label_list, output_dict=False)

        return {'score_dict': classification_score_dict, 'score_str': classification_score_str}


    # def bert_eval_with_noisy_labels(self, args, logger, t_loader, model, device, fast_mode):
    #     all_preds = []
    #     all_y = []
    #     model.eval()
    #
    #     if fast_mode:
    #         n_batch = len(t_loader)/10
    #
    #     with torch.no_grad():
    #         for idx, t_batch in enumerate(t_loader):
    #             input_ids, attention_mask, _, c_targets, _ = self.extract_bert_input_with_c_targets(t_batch, tag='org')
    #             input_ids = input_ids.to(device)
    #             attention_mask = attention_mask.to(device)
    #             c_targets = c_targets.to(device)
    #             n_targets_one_hot = t_batch['n_labels_one_hot'].to(device)
    #
    #             y_pred = model(input_ids, attention_mask, n_targets_one_hot)['logits']
    #             predicted = torch.max(y_pred.cpu(), 1)[1]
    #             all_preds.extend(predicted.numpy())
    #             all_y.extend(list(c_targets.cpu()))
    #             if fast_mode and idx > n_batch:
    #                 break
    #
    #         classification_score_dict = classification_report(all_y, np.array(all_preds).flatten(),
    #                                                           target_names=self.label_list, output_dict=True)
    #         classification_score_str = classification_report(all_y, np.array(all_preds).flatten(),
    #                                                          target_names=self.label_list, output_dict=False)
    #
    #     return {'score_dict': classification_score_dict, 'score_str': classification_score_str}


    def eval_model_with_both_labels(self, model, v_loader, device, fast_mode):
        all_preds = []
        all_y_c = []
        all_y_n = []
        model.eval()
        c_val_loss_sum = 0
        n_val_loss_sum = 0
        loss_fn = torch.nn.CrossEntropyLoss()

        if fast_mode:
            n_batch = len(v_loader)/10

        with torch.no_grad():
            for idx, t_batch in enumerate(v_loader):
                input_ids = t_batch['input_ids'].to(device)
                attention_mask = t_batch['attention_mask'].to(device)
                c_labels = t_batch['c_labels'].to(device)
                n_labels = t_batch['n_labels'].to(device)

                y_pred = model(input_ids, attention_mask)['logits']
                predicted = torch.max(y_pred.cpu(), 1)[1]

                c_val_loss_sum += loss_fn(y_pred, c_labels).item()
                n_val_loss_sum += loss_fn(y_pred, n_labels).item()

                all_preds.extend(predicted.numpy())
                all_y_c.extend(list(c_labels.cpu()))
                all_y_n.extend(list(n_labels.cpu()))


                if fast_mode and idx > n_batch:
                    break

            num_val_samples = len(all_y_c)

            classification_score_dict_n = classification_report(all_y_n, np.array(all_preds).flatten(),
                                                              target_names=self.label_list, output_dict=True)
            classification_score_str_n = classification_report(all_y_n, np.array(all_preds).flatten(),
                                                             target_names=self.label_list, output_dict=False)

            classification_score_dict_c = classification_report(all_y_c, np.array(all_preds).flatten(),
                                                              target_names=self.label_list, output_dict=True)
            classification_score_str_c = classification_report(all_y_c, np.array(all_preds).flatten(),
                                                             target_names=self.label_list, output_dict=False)

            c_val_loss_avg = c_val_loss_sum/num_val_samples
            n_val_loss_avg = n_val_loss_sum/num_val_samples

        return {'score_dict_n': classification_score_dict_n,
                'score_str_n': classification_score_str_n,
                'score_dict_c': classification_score_dict_c,
                'score_str_c': classification_score_str_c,
                'val_c_loss': c_val_loss_avg,
                'val_n_loss': n_val_loss_avg}

    def bert_val_eval_with_noisy_labels(self, model, v_loader, device, fast_mode):
        all_preds = []
        all_y_c = []
        all_y_n = []
        model.eval()
        c_val_loss_sum = 0
        n_val_loss_sum = 0
        loss_fn = torch.nn.CrossEntropyLoss()


        if fast_mode:
            n_batch = len(v_loader)/10

        with torch.no_grad():
            for idx, t_batch in enumerate(v_loader):
                input_ids, attention_mask, n_labels, c_labels, _ = \
                    self.extract_bert_input_with_c_targets(t_batch, tag='org')

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                c_labels = c_labels.to(device)
                n_labels = n_labels.to(device)

                n_labels_one_hot = t_batch['n_labels_one_hot'].to(device)

                y_pred = model(input_ids, attention_mask, n_labels_one_hot)['logits']
                predicted = torch.max(y_pred.cpu(), 1)[1]

                c_val_loss_sum += loss_fn(y_pred, c_labels).item()
                n_val_loss_sum += loss_fn(y_pred, n_labels).item()

                all_preds.extend(predicted.numpy())
                all_y_c.extend(list(c_labels.cpu()))
                all_y_n.extend(list(n_labels.cpu()))


                if fast_mode and idx > n_batch:
                    break

            num_val_samples = len(all_y_c)

            classification_score_dict_n = classification_report(all_y_n, np.array(all_preds).flatten(),
                                                              target_names=self.label_list, output_dict=True)
            classification_score_str_n = classification_report(all_y_n, np.array(all_preds).flatten(),
                                                             target_names=self.label_list, output_dict=False)

            classification_score_dict_c = classification_report(all_y_c, np.array(all_preds).flatten(),
                                                              target_names=self.label_list, output_dict=True)
            classification_score_str_c = classification_report(all_y_c, np.array(all_preds).flatten(),
                                                             target_names=self.label_list, output_dict=False)



            # sub_acc = self.get_sub_acc(all_y, all_preds, 10)
            # print(sub_acc)

        return {'score_dict_n': classification_score_dict_n,
                'score_str_n': classification_score_str_n,
                'score_dict_c': classification_score_dict_c,
                'score_str_c': classification_score_str_c,
                'val_c_loss': c_val_loss_sum/num_val_samples,
                'val_n_loss': n_val_loss_sum/num_val_samples}

    def save_emb_for_clustering(self, model, data_loader, global_step,
                                log_dir, writer, device):
        all_preds = []
        all_y = []
        model.eval()
        # save_dir = os.path.join(log_dir, 'bert_embeddings')
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        emb_list = []
        label_list = []

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                input_ids, attention_mask, labels = self.extract_bert_input(batch, tag='org')
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                embs = model.bert(input_ids, attention_mask)['pooler_output'].detach().cpu().numpy()
                emb_list.append(embs)
                label_list.append(labels.numpy())

        emb_array = np.vstack(emb_list)
        label_array = np.hstack(np.hstack(label_list))
        writer.add_embedding(emb_array,
                             metadata=label_array,
                             global_step=global_step,
                             tag=f'word embedding')


    # def extract_bert_input(self, data_batch, return_idx):
    #     input_ids = data_batch[f'input_ids']
    #     attention_mask = data_batch[f'attention_mask']
    #     labels = data_batch[f'labels']
    #     if return_idx:
    #         return input_ids, attention_mask, labels, data_batch[f'index']
    #     else:
    #         return input_ids, attention_mask, labels

    def extract_bert_input_with_purity(self, data_batch, tag):
        input_ids = data_batch[f'input_ids_{tag}']
        attention_mask = data_batch[f'attention_mask_{tag}']
        labels = data_batch[f'labels']
        purity = data_batch['purity']
        return input_ids, attention_mask, labels, purity, data_batch[f'index']

    # def extract_bert_input_with_c_targets(self, data_batch, tag):
    #     input_ids = data_batch[f'input_ids_{tag}']
    #     attention_mask = data_batch[f'attention_mask_{tag}']
    #     labels = data_batch[f'labels']
    #     c_targets = data_batch['c_targets']
    #     return input_ids, attention_mask, labels, c_targets, data_batch[f'index']

    def get_optimizer_grouped_parameters(self, args, model):
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        return optimizer_parameters

    def test(self, model, run=None):
        model.eval()

        preds = []
        gt_labels = []

        for xs, c_labs, _, _ in self.t_loader:
            xs = xs.to(self.device)
            c_labs = c_labs.to(self.device)

            pred = torch.argmax(model(xs), dim=1)
            preds.append(pred.cpu().numpy())
            gt_labels.append(c_labs.cpu().numpy())

        y_true = np.concatenate(gt_labels)
        y_pred = np.concatenate(preds)

        precision, recall, fscore, support = pr_score(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred)
        acc_sep = matrix.diagonal() / matrix.sum(axis=1)
        acc_total = np.sum(matrix.diagonal())/np.sum(matrix)

        if self.sw is not None and run is not None:
            bins = np.bincount(y_pred)
            if len(bins) != self.num_labels:
                assert(len(bins) < self.num_labels)
                zero_patch = np.zeros(self.num_labels - len(bins))
                bins = np.concatenate((bins, zero_patch))
            for l in range(self.num_labels):
                self.sw.add_scalar(f'num_pred{l}', bins[l], run)


            # mat3channel = np.repeat(matrix[None, ...], 3, axis=0)
            # self.sw.add_image('pred_matrix', mat3channel, run)

        return acc_sep, acc_total, precision, recall


    def get_trainer_info(self):
        pass


    def load_pre_trained(self, args, model):
        if args.tf_model_path != '':
            raise ValueError("We don't supoort TF loading yet")
        assert args.py_model_path != ''
        model.load_pretrained_lm_model(args, args.py_model_path)