# https://github.com/yihong-chen/neural-collaborative-filtering/blob/master/src/gmf.py

import torch

from model.embedding import EmbeddingGrad


class GMF(torch.nn.Module):
    def __init__(self, n_users, n_items, h_dim_size, use_cuda, use_logit=False):
        super(GMF, self).__init__()
        self.num_users = n_users
        self.num_items = n_items
        self.latent_dim = h_dim_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.use_cuda = use_cuda
        self.use_logit = use_logit
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding = EmbeddingGrad(num_embedding=self.num_items, embedding_dim=self.latent_dim)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        if use_cuda:
            self = self.cuda()

    def forward(self, user_indices, item_indices):

        user_embedding = self.embedding_user(user_indices).squeeze()
        item_embedding = self.embedding(item_indices)

        if item_embedding.ndim == 3:

            user_embedding = user_embedding.unsqueeze(1).repeat(1, item_embedding.shape[1], 1)

        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)

        if self.use_logit:
            logits = self.logistic(logits)

        return logits


    def init_weight(self):
        pass
