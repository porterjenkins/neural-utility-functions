import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import numpy as np
from model.embedding import EmbeddingGrad
import torch


class UtilityEncoder(nn.Module):

    def __init__(self, n_items, h_dim_size, use_cuda=False):
        super(UtilityEncoder, self).__init__()
        self.n_items = n_items
        self.h_dim_size = h_dim_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.use_cuda = use_cuda


        self.embedding = EmbeddingGrad(n_items, h_dim_size)
        self.weights = nn.Linear(h_dim_size, 1)

        if use_cuda:
            self = self.cuda()


    def forward(self, users, items):
        e_i = self.embedding(items)
        y_hat = self.weights(e_i)

        return y_hat


    def get_embedding_mtx(self):

        return np.transpose(self.embedding.weights.weight.data.numpy())


