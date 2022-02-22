# https://github.com/EthanRosenthal/torchmf/blob/master/torchmf.py


import torch
from torch import nn
from model.embedding import EmbeddingGrad

class MatrixFactorization(nn.Module):
    """
    Base module for explicit matrix factorization.
    """

    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False,
                 use_logit=False,
                 use_cuda=False):
        """
        Parameters
        ----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super(MatrixFactorization, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = EmbeddingGrad(n_items, 1)
        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_embeddings = EmbeddingGrad(num_embedding=self.n_items, embedding_dim=self.n_factors)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.logistic = torch.nn.Sigmoid()
        self.sparse = sparse
        self.use_logit = use_logit
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        if use_cuda:
            self = self.cuda()

    def forward(self, users, items):
        """
        Forward pass through the model. For a single user and item, this
        looks like:
        user_bias + item_bias + user_embeddings.dot(item_embeddings)
        Parameters
        ----------
        users : np.ndarray
            Array of user indices
        items : np.ndarray
            Array of item indices
        Returns
        -------
        preds : np.ndarray
            Predicted ratings.
        """
        ues = self.user_embeddings(users).squeeze()
        uis = self.item_embeddings(items)

        b_user =  self.user_biases(users).view(-1, 1)
        b_item = self.item_biases(items)

        if uis.ndim == 3:
            b_user = b_user.repeat(1, uis.shape[1]).unsqueeze(2)
            ues = ues.unsqueeze(1).repeat(1, uis.shape[1], 1)

        preds = b_user + b_item + (self.dropout(ues) * self.dropout(uis)).sum(dim=-1, keepdim=True)
        #preds = (self.dropout(ues) * self.dropout(uis)).sum(dim=-1, keepdim=True)

        if self.use_logit:
            preds = self.logistic(preds)

        return preds

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users, items):
        return self.forward(users, items)


def bpr_loss(preds, vals):
    sig = nn.Sigmoid()
    return (1.0 - sig(preds)).pow(2).sum()