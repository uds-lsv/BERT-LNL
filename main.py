import argparse
from loading_utils import prepare_data, prepare_af_data
from utils import create_logger, save_args, create_trainer, load_config, save_config
import numpy as np
import torch
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='IMDB', choices=['SST-2', 'AG_News',
                                                                        'TREC', 'IMDB', 'Yelp-5',
                                                                        'Yoruba', 'Hausa'])
    parser.add_argument('--data_root', type=str, default="")
    parser.add_argument('--log_root', type=str, default="",
                        help='output directory to save logs in training/testing')

    parser.add_argument('--trainer_name', type=str, default='bert_wn',
                        choices=['bert_wn', 'bert_ct', 'bert_cm', 'bert_cmgt', 'bert_smoothing'],
                        help='trainer selection: '
                             'bert_wn: without noise-handling,'
                             'bert_ct: co-teaching, '
                             'bert_cm: noise matrix, '
                             'bert_cmgt: ground truth noise matrix,'
                             'bert_smoothing: label smoothing')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        choices=['bert-base-uncased', 'bert-base-cased',
                                 'bert-large-uncased', 'bert-base-multilingual-cased'],
                        help='backbone selection')
    parser.add_argument('--exp_name', type=str, default='')


    # Preprocessing Related
    parser.add_argument('--max_sen_len', type=int, default=512,
                        help='max sentence length, longer sentences will be truncated')
    parser.add_argument('--special_token_offsets', type=int, default=2,
                        help='number of special tokens used in bert tokenizer for text classification')
    parser.add_argument('--truncate_mode', type=str, default='last',
                        choices=['hybrid, last'], help='last: last 510 tokens, hybrid: first 128 + last 382')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='freeze the bert backbone, i.e. use bert as feature extractor')

    # BERT settings Related
    parser.add_argument('--bert_dropout_rate', type=float, default=0.1)
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='noise level for injected noise')
    parser.add_argument('--noise_type', default='uniform_m',
                        choices=['uniform_m', 'sflip'],
                        help='noise types: uniform_m: uniform noise, sflip: single-flip noise')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='if no validation set is provided, use this fraction of training set as validation set')

    # training related
    parser.add_argument('--num_epochs', type=int, default=1, help='set either num_epochs or num_training_steps')
    parser.add_argument('--num_training_steps', type=int, default=-1, help='set it to -1 if num_epochs is set')
    parser.add_argument('--train_eval_freq', type=int, default=10,
                        help='evaluate the model on training set after every [train_eval_freq] training steps')
    parser.add_argument('--eval_freq', type=int, default=50,
                        help='evaluate the model on the validation and test sets'
                             'after every [eval_freq] training steps')
    parser.add_argument('--fast_eval', action='store_true',
                        help='use 10% of the test set for evaluation, to speed up the evaluation prcoess')

    parser.add_argument('--nl_batch_size', type=int, default=16,
                        help='noisy labeled samples per batch, can be understood as the training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=50,
                        help='evaluation batch size during testing')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='warmup steps for learning rate scheduler')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--gen_val', action='store_true',
                        help='generate validation set, enable it if there is no validation set')
    parser.add_argument('--store_model', type=int, default=0, help='save models after training')


    # co-teaching related
    parser.add_argument('--forget_factor', type=float, default=1.0)
    parser.add_argument('--T_k', type=int, default=10)
    parser.add_argument('--c', type=float, default=1.0)


    # smoothing trainer related
    parser.add_argument('--smoothing_factor', type=float, default=0.2,
                        help='label smoothing levels, 0.0 means no smoothing')


    # cm trainer related
    parser.add_argument('--cm_mse_weight', type=float, default=0.01,
                        help='regularization factor for the mse loss, check https://aclanthology.org/N19-1328/')


    # optimizer related
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--exp_decay_rate', type=float, default=0.9998)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--patience', type=float, default=20, help='patience for early stopping')

    # hardware related
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cuda_device', type=str, default="0")
    parser.add_argument('--manualSeed', type=int, default=1234, help='random seed for reproducibility')
    parser.add_argument('--noisy_label_seed', type=int, default=1234, help='random seed for reproducibility')

    args = parser.parse_args()

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.backends.cudnn.benchmark = False
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.deterministic = True

    # Create the Handler for logging records/messages to a file
    logger, log_dir = create_logger(args.log_root, args)
    save_args(log_dir, args)
    logger.info("Training started")
    num_classes_map = {'AG_News': 4, 'TREC': 6, 'SST-2': 2, 'IMDB': 2, 'Yelp-5': 5,
                       'Yoruba':7, 'Hausa':5}

    logger.info(f'log dir: {log_dir}')
    num_classes = num_classes_map[args.dataset]
    r_state = np.random.RandomState(args.noisy_label_seed)

    if args.dataset in ['SST-2', 'AG_News', 'TREC', 'IMDB', 'Yelp-5']:
        logger.info(f'loading {args.dataset}...')


        if args.dataset in ['SST-2', 'AG_News', 'TREC', 'Yelp-5']:
            has_val = False
            has_ul = False
        elif args.dataset in ['IMDB']:
            has_val = True
            has_ul = True
        else:
            raise ValueError('need to set has_val and has_ul')

        nl_set, ul_set, v_set, t_set, l2id, id2l = prepare_data(args, logger, r_state, num_classes, has_val, has_ul)
    elif args.dataset in ['Yoruba', 'Hausa']:
        has_ul=False
        nl_set, ul_set, v_set, t_set, l2id, id2l = prepare_af_data(args, logger,
                                                                   num_classes, has_ul)
    else:
        raise NotImplementedError(f"dataset {args.dataset} not supported")


    model_config = load_config(args)
    model_config['num_classes'] = num_classes

    trainer = create_trainer(args, logger, log_dir, model_config, (nl_set, ul_set, v_set, t_set, l2id, id2l), r_state)
    trainer.train(args, logger, (nl_set, ul_set, v_set, t_set, l2id, id2l))
    save_config(log_dir, 'model_config', trainer.model_config)  # model_config could be updated during model creation


if __name__=='__main__':
    main()