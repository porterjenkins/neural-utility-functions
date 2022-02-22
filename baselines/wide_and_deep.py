import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embedding import EmbeddingGrad



class WideAndDeepPretrained(nn.Module):

    def __init__(self, n_items, h_dim_size, wide, wide_dim, fc1=64, fc2=32,):
        super(WideAndDeepPretrained, self).__init__()
        self.n_items = n_items
        self.h_dim_size = h_dim_size


        self.embedding = EmbeddingGrad(n_items, h_dim_size)
        self.fc_1 = nn.Linear(h_dim_size, fc1)
        self.fc_2 = nn.Linear(fc1, fc2)

        self.wide = EmbeddingGrad(n_items, wide_dim, init_embed=wide)



        self.output_layer = nn.Linear(wide_dim + fc2, 1)

    def get_input_grad(self, indices):
        """
        Get gradients with respect to inputs
        :param indices: (ndarray) array of item indices
        :return: (tensor) tensor of gradients with respect to inputs
        """
        if indices.ndim == 1:
            indices = indices.reshape(-1, 1)


        dims = [d for d in indices.shape] + [1]
        idx_tensor = torch.LongTensor(indices).reshape(dims)

        grad = self.embedding.get_grad(indices)
        grad_at_idx = torch.gather(grad, -1, idx_tensor)
        return torch.squeeze(grad_at_idx)


    def _forward_set(self, x):
        h = self.embedding(x)
        h = F.relu(self.fc_1(h))
        h = F.relu(self.fc_2(h))

        wide = self.wide(x)
        h = torch.cat([h, wide], dim=-1)

        y_hat = self.output_layer(h)
        return y_hat


    def forward(self, x, x_c=None, x_s=None):

        y_hat = self._forward_set(x)

        if x_c is not None and x_s is not None:

            y_hat_c = self._forward_set(x_c)
            y_hat_s = self._forward_set(x_s)

            return y_hat, torch.squeeze(y_hat_c), torch.squeeze(y_hat_s)

        else:

            return y_hat


    def fit(self, X_train, y_train, batch_size, k, lr, n_epochs, loss_step, eps):
        pass


    def predict(self, X_test):
        pass

class WideAndDeep(nn.Module):

    def __init__(self, n_items, h_dim_size, fc1=64, fc2=32, use_cuda=False, use_embedding=True, use_logit=False):
        super(WideAndDeep, self).__init__()
        self.n_items = n_items
        self.h_dim_size = h_dim_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.use_cuda = use_cuda
        self.use_logit = use_logit
        self.use_embedding = use_embedding
        self.embedding = EmbeddingGrad(n_items, h_dim_size, use_cuda=use_cuda)
        self.fc_1 = nn.Linear(h_dim_size, fc1)
        self.fc_2 = nn.Linear(fc1, fc2)
        self.output_layer = nn.Linear(n_items + fc2, 1)
        self.logistic = torch.nn.Sigmoid()


        if use_cuda:
            self = self.cuda()



    def forward(self, users, items):

        h = self.embedding(items)
        h = F.relu(self.fc_1(h))
        h = F.relu(self.fc_2(h))
        h = torch.cat([h, items], dim=-1)

        y_hat = self.output_layer(h)

        if self.use_logit:
            y_hat = self.logistic(y_hat)

        return y_hat



