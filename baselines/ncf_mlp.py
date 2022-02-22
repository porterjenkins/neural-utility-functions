# https://github.com/yihong-chen/neural-collaborative-filtering/blob/master/src/mlp.py

import torch
from model.embedding import EmbeddingGrad


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.use_cuda = config['use_cuda']
        self.use_logit = config["use_logit"]
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding = EmbeddingGrad(num_embedding=self.num_items, embedding_dim=self.latent_dim)


        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        if self.use_cuda:
            self = self.cuda()

    def forward(self, user_indices, item_indices):

        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding(item_indices)

        if item_embedding.ndim == 2:
            item_embedding = item_embedding.unsqueeze(1)

        # align user and item dimensions for concatenation for supp/comp sets
        elif item_embedding.ndim == 3:
            user_embedding = user_embedding.repeat(1, item_embedding.shape[1], 1)

        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)
        y_hat = self.affine_output(vector)
        if self.use_logit:
            y_hat = self.logistic(y_hat).squeeze(dim=-1)

        return y_hat

