import torch
import torch.nn as nn
from noise_functions import get_uniform_m_flip_mat, get_single_flip_mat

class CM(nn.Module):

    def __init__(self, model_config, args, base_model):
        super(CM, self).__init__()

        self.base_model = base_model
        num_classes = model_config['num_classes']
        # self.noise_matrix = nn.Parameter(torch.log(torch.eye(num_classes) + 1e-6), requires_grad=True)
        self.noise_model = nn.Parameter(torch.eye(num_classes), requires_grad=True)
        self.mat_normalizer = nn.Softmax(dim=1)
        self.logits2dist = nn.Softmax(dim=1)

    def forward(self, x, x_length):
        clean_logits = self.base_model(x, x_length)['logits']
        clean_dist = self.logits2dist(clean_logits)
        # trans_mat = self.mat_normalizer(self.noise_model)
        logits = torch.matmul(clean_dist, self.noise_model)

        return {'logits': logits}


class CMGT(nn.Module):
    def __init__(self, model_config, args, base_model, noise_mat):
        super(CMGT, self).__init__()

        self.base_model = base_model
        self.num_classes = model_config['num_classes']
        self.noise_type = args.noise_type
        self.noise_level = args.noise_level
        if noise_mat is None:
            mat = self.get_matrix()
        else:
            mat = noise_mat
        self.noise_matrix = nn.Parameter(torch.tensor(mat).float(), requires_grad=False)
        self.logits2dist = nn.Softmax(dim=1)

    def get_matrix(self):
        if self.noise_type == 'sflip':
            mat = get_single_flip_mat(self.noise_level, self.num_classes)
        elif self.noise_type == 'uniform':
            raise NotImplementedError('noise type not supported')
        elif self.noise_type == 'uniform_m':
            mat = get_uniform_m_flip_mat(self.noise_level, self.num_classes)
        else:
            raise NotImplementedError('noise type not supported')

        return mat

    def forward(self, x, x_length):
        clean_logits = self.base_model(x, x_length)['logits']
        clean_dist = self.logits2dist(clean_logits)
        noisy_prob = torch.matmul(clean_dist, self.noise_matrix)
        log_noisy_logits = torch.log(noisy_prob + 1e-6)

        return {'log_noisy_logits': log_noisy_logits}