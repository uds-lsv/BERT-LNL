# BERT_LNL
Code for paper ["Is BERT Robust to Label Noise? A Study on Learning with Noisy Labels in
Text Classification"](https://aclanthology.org/2022.insights-1.8.pdf).

## Data Preparation
### Datasets
We run our experiments on the following four datasets: AG-News, IMDB, Yorùbá, and Hausa.

| Name | # train | # validation | # test | data source                                            |
|:--------|:--------|:-------------|:-------|:-------------------------------------------------------|
| AGNews | 108000  | 12000        | 7600   | [LINK](https://arxiv.org/abs/1509.01626)               | 
| IMDB | 21246   | 3754         | 2500   | [LINK](https://dl.acm.org/doi/10.5555/2002472.2002491) | 
| Yorùbá | 1340    | 189         | 379   | [LINK](https://github.com/uds-lsv/transfer-distant-transformer-african) | 
| Hausa | 2045    | 290         | 582   | [LINK](https://github.com/uds-lsv/transfer-distant-transformer-african) | 

### Preprocessing
Create a `data` directory as the home directory for all datasets, then create a folder for each dataset in the `data` directory.

For each dataset, create the following files:
- `[train/validation/test].txt`: one document per line.
- `[train/validation/test]_labels.pickle` files: list of labels (we assume that labels are already encoded in label ids, e.g., [0,0,1,2,3,...,4]).
- If there is no validation data for a given dataset, then there is no need to create the corresponding `validation` files.
- Save files in a `txt_data` directory inside the dataset folder. For example, the full path to the `train.txt` file of the AGNews dataset is `data/AGNews/txt_data/train.txt`.

## Examples
Run `BERT-WN` on AG-News, with 20% single-flip label noise:
```
CUDA_VISIBLE_DEVICES=[CUDA_ID] python3 ../main.py \
--dataset AG-NEWs \
--log_root [LOG_ROOT] \
--data_root [DATA_ROOT] \
--trainer_name bert_wn \
--model_name bert-base-uncased \
--gen_val \
--nl_batch_size 32 \
--eval_batch_size 32 \
--gradient_accumulation_steps 1 \
--max_sen_len 64 \
--lr 0.00002 \
--num_training_steps 3000 \
--patience 25 \
--eval_freq 50 \
--store_model $STORE_MODEL \
--noise_level 0.2 \
--noise_type sflip \
--manualSeed 1234
```

