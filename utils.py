import yaml
import os
import torch.nn.functional as F
import logging
import json
import pickle
import datetime
from trainers.bert_wn_trainer import BertWN_Trainer
from trainers.bert_ct_trainer import BertCT_Trainer
from trainers.bert_cm_trainer import BertCM_Trainer
from trainers.bert_cmgt_trainer import BertCMGT_Trainer
from trainers.bert_smoothing_trainer import BertSmoothing_Trainer



def create_trainer(args, logger, log_dir, model_config, full_dataset, random_state):
    if args.trainer_name == 'bert_wn':
        trainer = BertWN_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    elif args.trainer_name == 'bert_ct':
        trainer = BertCT_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    elif args.trainer_name == 'bert_cm':
        trainer = BertCM_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    elif args.trainer_name == 'bert_cmgt':
        trainer = BertCMGT_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    elif args.trainer_name == 'bert_smoothing':
        trainer = BertSmoothing_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    else:
        raise NotImplementedError('Unknown Trainer Name')

    return trainer




def load_config(args):
    model_config = {}
    model_config['drop_rate'] = args.bert_dropout_rate
    return model_config

def save_config(save_dir, config_name, config_data):
    save_path = os.path.join(save_dir, f'{config_name}.yaml')

    with open(save_path, 'w') as file:
        yaml.dump(config_data, file)

def pickle_save(file_name, data):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_name):
    with open(file_name, 'rb') as handle:
        b = pickle.load(handle)
    return b


def create_log_path(log_root, args, starting_time):
    staring_time_str = starting_time.strftime("%m_%d_%H_%M_%S")
    suffix = staring_time_str

    suffix += f'_{args.noise_type}_nle{args.noise_level}'
    suffix += f'_nlb{args.nl_batch_size}'

    log_dir = os.path.join(log_root, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, 'log.txt')
    return log_path, log_dir


def create_logger(log_root, args):
    starting_time = datetime.datetime.now()

    log_path, log_dir = create_log_path(log_root, args, starting_time)

    # check if the file exist

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_path)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger, log_dir


def save_args(log_dir, args):
    arg_save_path = os.path.join(log_dir, 'config.json')
    with open(arg_save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)