import copy
import torch
from torch.utils import data
import numpy as np
import torch.nn.functional as F
from noise_functions import make_data_noisy


class TextBertDataset(data.Dataset):
    def __init__(self, args, input_data, tokenizer, r_state, num_classes, make_noisy=False):
        # https://huggingface.co/transformers/custom_datasets.html
        self.args=args
        self.tokenizer = tokenizer
        self.encodings = input_data['features']
        self.clean_labels = input_data['labels']
        self.text = input_data['text']
        self.make_noisy = make_noisy
        self.num_classes = num_classes
        self.noisy_labels = None
        self.purity_list = None


        if make_noisy:
            clean_labels_copy = copy.deepcopy(input_data['labels'])
            nl_y = make_data_noisy(clean_labels_copy, args.noise_level, noise_type=args.noise_type, r_state=r_state,
                                   num_classes=self.num_classes)
            self.noisy_labels = nl_y
            self.purity_list = torch.tensor((np.array(nl_y) == np.array(clean_labels_copy))).long()
        else:
            self.noisy_labels = -1 * torch.ones(len(input_data['labels'])).long()
            self.purity_list = -1 * torch.ones(len(input_data['labels'])).long()

    def get_subset_by_indices(self, indices):
        sub_encodings = {key: val[indices] for key, val in self.encodings.items()}
        sub_text = self.text[indices]
        sub_labels = self.clean_labels[indices]
        input_data = {'data':sub_encodings, 'labels':sub_labels, 'text': sub_text}

        # set make_noisy to False here, because we later manually add the noisy labels to the subset
        subdataset = TextBertDataset(self.args, input_data, self.tokenizer, r_state=None,
                                     num_classes=self.num_classes, make_noisy=False)

        subdataset.noisy_labels = self.noisy_labels[indices]
        subdataset.purity_list = self.purity_list[indices]

        return subdataset

    def __len__(self):
        return len(self.clean_labels)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item['c_labels'] = self.clean_labels[index]
        item['n_labels'] = self.noisy_labels[index]
        item['purity'] = self.purity_list[index]
        item['index'] = index
        return item



class TextBertRealDataset(data.Dataset):
    def __init__(self, args, input_data, noisy_labels, tokenizer, num_classes):
        # The difference to TextBertDataset is that the noisy labels are provided directly in RealDataset
        self.args = args
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.text = input_data['text']
        self.encodings = input_data['features']
        self.clean_labels = input_data['labels']
        self.noisy_labels = noisy_labels
        assert len(self.noisy_labels) == len(self.clean_labels)
        self.purity_list = torch.tensor((np.array(self.clean_labels) == np.array(self.noisy_labels))).long()

        self.noisy_labels_one_hot = F.one_hot(torch.tensor(self.noisy_labels))

    def get_noise_mat(self):
        assert self.clean_labels is not None, "noise matrix unavailable as no clean data is available"
        noise_mat = np.zeros((self.num_classes, self.num_classes))
        for i, j in zip(self.clean_labels, self.noisy_labels):
            noise_mat[i, j] += 1
        noise_mat = noise_mat/noise_mat.sum(axis=1)[:, None]
        return noise_mat

    def get_subset_by_indices(self, indices):
        sub_encodings = {key: val[indices] for key, val in self.encodings.items()}

        sub_text = self.text[indices]
        sub_labels = self.clean_labels[indices]
        input_data = {'data': sub_encodings, 'labels': sub_labels, 'text': sub_text}
        sub_noisy_labels = self.noisy_labels[indices]
        return TextBertRealDataset(self.args, input_data, sub_noisy_labels, self.tokenizer, self.num_classes)


    def __len__(self):
        return len(self.clean_labels)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item['c_labels'] = self.clean_labels[index]
        item['n_labels'] = self.noisy_labels[index]
        item['purity'] = self.purity_list[index]
        item['index'] = index

        item['n_labels_one_hot'] = self.noisy_labels_one_hot[index]
        return item