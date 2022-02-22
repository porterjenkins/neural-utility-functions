import numpy as np
import torch
import torch.nn as nn


class EmbeddingGrad(nn.Module):

    def __init__(self, num_embedding, embedding_dim, init_embed=None, use_cuda=False):
        super(EmbeddingGrad, self).__init__()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.use_cuda = use_cuda

        self.weights = nn.Linear(num_embedding, embedding_dim)

        if init_embed is not None:
            self.weights.weight = torch.nn.Parameter(init_embed)


    def init_embed(self):
        pass


    def forward(self, ones):
        e = self.weights(ones)
        return e

