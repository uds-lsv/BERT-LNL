import os
import copy
import numpy as np
import pickle
import torch
import wandb
from tqdm import tqdm
from text_dataset import TextBertDataset, TextBertRealDataset
from transformers import AutoTokenizer
from transformers import BertTokenizerFast, BertTokenizer
import utils


def prepare_data(args, logger, r_state, num_classes, has_val, has_ul):
    # used for experiments with injected noise

    tokenizer = load_tokenizer(args)
    tr_data, val_data = get_training_validation_set(args, logger, tokenizer, r_state, has_val, num_classes)
    test_data = load_and_cache_text(args, tokenizer, logger, tag='test')

    n_set = TextBertDataset(args, tr_data, tokenizer, r_state, num_classes, make_noisy=True)
    v_set = TextBertDataset(args, val_data, tokenizer, r_state, num_classes, make_noisy=True)
    t_set = TextBertDataset(args, test_data, tokenizer, r_state, num_classes, make_noisy=False)

    n_set_noisy_labels = copy.deepcopy(n_set.noisy_labels)
    v_set_noisy_labels = copy.deepcopy(v_set.noisy_labels)
    n_set_noisy_labels_hash = hash(tuple(n_set_noisy_labels))
    v_set_noisy_labels_hash = hash(tuple(v_set_noisy_labels))
    # wandb.run.summary["train_n_hash"] = n_set_noisy_labels_hash
    # wandb.run.summary["val_n_hash"] = v_set_noisy_labels_hash

    u_set = None
    l2id = None
    id2l = None

    return n_set, u_set, v_set, t_set, l2id, id2l


def prepare_af_data(args, logger, num_classes, has_ul):
    tokenizer = load_tokenizer(args)
    n_set = get_clean_and_noisy_data_by_tag(args, logger, tokenizer, num_classes, tag='train')
    v_set = get_clean_and_noisy_data_by_tag(args, logger, tokenizer, num_classes, tag='validation')
    t_set = get_clean_and_noisy_data_by_tag(args, logger, tokenizer, num_classes, tag='test')

    assert not has_ul  # we do not have unlabeled data in Yoruba and Hausa dataset
    u_set = None

    label_mapping_data_dir = os.path.join(args.data_root, args.dataset, 'txt_data')
    l2id = utils.pickle_load(os.path.join(label_mapping_data_dir, 'l2idx.pickle'))
    id2l = utils.pickle_load(os.path.join(label_mapping_data_dir, 'idx2l.pickle'))

    return n_set, u_set, v_set, t_set, l2id, id2l


def get_training_validation_set(args, logger, tokenizer, r_state, has_val, num_classes):
    # sanity check: args.gen_val is used when there is no validation set
    if has_val:
        assert not args.gen_val

    tr_data = load_and_cache_text(args, tokenizer, logger, tag='train')

    if has_val:  # original validation set available
        val_data = load_and_cache_text(args, tokenizer, logger, tag='validation')
    elif args.gen_val:  # create validation set using the training set
        val_indices_path = os.path.join(args.data_root, args.dataset, 'val_indices', f'{args.dataset}_val_indices.pickle')
        with open(val_indices_path, 'rb') as handle:
            val_indices = pickle.load(handle)

        val_mask = np.zeros(len(tr_data['labels']), dtype=bool)
        val_mask[val_indices] = True

        val_features = {k: v[val_mask] for k,v in tr_data['features'].items()}
        val_labels = tr_data['labels'][val_mask]
        val_text  = np.array(tr_data['text'])[val_mask]

        train_features = {k: v[~val_mask] for k,v in tr_data['features'].items()}
        train_labels = tr_data['labels'][~val_mask]
        train_text  = np.array(tr_data['text'])[~val_mask]

        val_data = {"features": val_features, "labels": val_labels, "text": val_text}
        tr_data = {"features": train_features, "labels": train_labels, "text": train_text}

    else:
        raise ValueError("we need a validation set, set gen_val to True to extract"
                         "a subset from the training data as validation data")

    return tr_data, val_data

def get_clean_and_noisy_data_by_tag(args, logger, tokenizer, num_classes, tag):
    noisy_data_tag = f'{tag}_clean'

    #get text data with noisy labels
    clean_noisy_data = load_and_cache_text(args, tokenizer, logger, tag=noisy_data_tag)

    #get the clean training and the clean validation sets
    txt_data_dir = os.path.join(args.data_root, args.dataset, 'txt_data')
    input_path = os.path.join(txt_data_dir, f'{tag}_clean_noisy_labels.pickle')
    noisy_labels = load_pickle_data(input_path)

    td = TextBertRealDataset(args, clean_noisy_data, noisy_labels, tokenizer, num_classes)
    return td



def load_and_cache_text(args, tokenizer, logger, tag):
    cached_features_dir = os.path.join(args.data_root, args.dataset, 'bert_preprocessed') # cache dir (output dir)
    txt_data_dir = os.path.join(args.data_root, args.dataset, 'txt_data') # input file dir

    if not os.path.exists(cached_features_dir):
        os.makedirs(cached_features_dir)

    cached_features_path = os.path.join(cached_features_dir,
                                        f'{tag}_trun_{args.truncate_mode}_maxl_{args.max_sen_len}')
    input_path = os.path.join(txt_data_dir, f'{tag}.txt')

    docs = read_txt(input_path)

    if os.path.exists(cached_features_path):
        logger.info(f'[Loading and Caching] loading from cache...')
        features = torch.load(cached_features_path)
    else:
        logger.info(f'[Loading and Caching] number of documents = {len(docs)}')
        logger.info(f'[Loading and Caching] convert text to features...')
        features = get_input_features(docs, tokenizer, args)
        logger.info("[Loading and Caching] saving/caching the features...")
        torch.save(features, cached_features_path)
        logger.info("[Loading and Caching] saved")

    logger.info(f'[Loading and Caching] loading labels...')
    input_path = os.path.join(txt_data_dir, f'{tag}_labels.pickle')
    with open(input_path, 'rb') as handle:
        labels = np.array(pickle.load(handle))

    return {"features": features, "labels": labels, "text": docs}




def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return tokenizer


def load_pickle_data(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data



def read_txt(file_path):
    text_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        text_list = [line.rstrip('\n') for line in f]
    return text_list


def truncate_token_ids(token_ids, args, limit):
    if args.truncate_mode == 'last':
        return token_ids[-limit:]
    elif args.truncate_mode == 'hybrid':
        return token_ids[:128] + token_ids[-382:]
    else:
        raise ValueError('truncate model not supported')


def get_input_features(docs, tokenizer, args):
    limit = args.max_sen_len - args.special_token_offsets
    # sanity check
    if args.truncate_mode == 'hybrid':
        assert args.max_sen_len == 512
    assert limit > 0
    num_docs = len(docs)

    input_id_tensor = torch.zeros((num_docs, args.max_sen_len)).long()
    length_tensor = torch.zeros(num_docs).long()
    token_type_tensor = torch.zeros((num_docs, args.max_sen_len)).long()
    attention_mask_tensor = torch.zeros((num_docs, args.max_sen_len)).long()

    for idx, doc in enumerate(tqdm(docs, desc='convert docs to tensors')):
        tokens = tokenizer.tokenize(doc)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids_trunc = truncate_token_ids(token_ids, args, limit)

        input_ids = torch.tensor([tokenizer.cls_token_id] + token_ids_trunc +
                                 [tokenizer.sep_token_id]).long()
        input_ids_length = len(input_ids)
        # token_types = torch.zeros(len(input_ids)).long()
        attention_mask = torch.ones(len(input_ids)).long()

        input_id_tensor[idx, :input_ids_length] = input_ids
        length_tensor[idx] = input_ids_length
        attention_mask_tensor[idx, :input_ids_length] = attention_mask

    return {'input_ids': input_id_tensor, 'token_type_ids': token_type_tensor,
            'attention_mask': attention_mask_tensor, 'length': length_tensor}