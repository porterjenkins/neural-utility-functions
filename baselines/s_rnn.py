import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import config.config as cfg
from generator.generator import SeqCoocurrenceGenerator
from preprocessing.utils import split_train_test_user, load_dict_output
from preprocessing.interactions import Interactions
import numpy as np
from model.embedding import EmbeddingGrad
from model._loss import mrs_loss, utility_loss, loss_mse


class SRNNTrainer(object):
    def __init__(self, srnn, data, params, use_cuda=False, use_utility_loss=False, user_item_rating_map=None,
                 item_rating_map=None, k=None, n_items=None):
        self.srnn = srnn
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.use_utility_loss = use_utility_loss
        self.data = data
        self.user_item_rating_map = user_item_rating_map
        self.item_rating_map = item_rating_map
        self.k = k
        self.batch_size = params['batch_size']
        self.n_items = n_items

        self.h_dim = params['h_dim']
        self.n_epochs = params['n_epochs']
        self.lr = params['lr']
        self.loss_step = params['loss_step']
        self.eps = params['eps']
        self.seq_len = params['seq_len']

        self.optimizer = optim.Adam(srnn.parameters(), lr=self.lr)

    def train(self):
        X = self.data[0]
        y = self.data[1]



        gen = self.generator(self.X_train, self.y_train)

        loss_arr = []
        iter = 0
        cum_loss = 0
        prev_loss = -1

        while gen.epoch_cntr < self.n_epochs:
            # print('Start Epoch #', gen.epoch_cntr)

            train_loss = self.do_epoch(gen)
            cum_loss += train_loss
            train_loss.backward()
            self.optimizer.step()

            if iter % self.loss_step == 0:
                if iter == 0:
                    avg_loss = cum_loss
                else:
                    avg_loss = cum_loss / self.loss_step
                print("iteration: {} - loss: {}".format(iter, avg_loss))
                cum_loss = 0

                loss_arr.append(avg_loss)

                if abs(prev_loss - train_loss) < self.eps:
                    print('early stopping criterion met. Finishing training')
                    print("{} --> {}".format(prev_loss, train_loss))
                    break
                else:
                    prev_loss = train_loss

            iter += 1

        # h = self.srnn.init_hidden()
        # y_hat, h = srnn.forward(X_test.values[:, 1], h)
        # rmse = np.sqrt(np.mean(np.power(y_test.values - y_hat.flatten().detach().numpy(), 2)))
        # print(rmse)

        return self.srnn

    def generator(self, X_train, y_train):

        return SeqCoocurrenceGenerator(X_train, y_train, batch_size=self.batch_size,
                                    user_item_rating_map=self.user_item_rating_map,
                                    item_rating_map=self.item_rating_map, shuffle=True,
                                    c_size=self.k, s_size=self.k, n_item=self.srnn.n_items,
                                    seq_len=self.seq_len)

    def do_epoch(self, gen):

        self.h_init = self.srnn.init_hidden()

        if self.use_utility_loss:
            x_batch, y_batch, x_c_batch, y_c, x_s_batch, y_s = gen.get_batch(as_tensor=True)
            x_batch = x_batch[:, 1:]
            self.optimizer.zero_grad()

            y_hat, h = self.srnn.forward(x_batch, self.h_init)

            x_c_batch = torch.transpose(x_c_batch, 0, 1)
            x_s_batch = torch.transpose(x_s_batch, 0, 1)

            set_dims = (self.seq_len, self.k, self.batch_size)

            y_hat_c = torch.zeros(set_dims)
            for i in range(x_c_batch.shape[0]):
                y_hat_c[i], h = self.srnn.forward(x_c_batch[i], self.h_init)

            y_hat_s = torch.zeros(set_dims)
            for i in range(x_c_batch.shape[0]):
                y_hat_s[i], h = self.srnn.forward(x_s_batch[i], self.h_init)

            y_hat_c = torch.transpose(y_hat_c, 0,1)
            y_hat_s = torch.transpose(y_hat_s, 0, 1)


            loss_u = utility_loss(y_hat, y_hat_c, y_hat_s, np.transpose(y_batch), np.transpose(y_c), np.transpose(y_s))
            loss_u.backward(retain_graph=True)

            x_grad = self.srnn.get_input_grad(x_batch)
            x_c_grad = self.srnn.get_input_grad(x_c_batch)
            x_s_grad = self.srnn.get_input_grad(x_s_batch)

            loss = mrs_loss(loss_u, x_grad.unsqueeze(2), torch.transpose(x_c_grad,0,1), torch.transpose(x_s_grad,0,1))

        else:
            x_batch, y_batch = gen.get_batch(as_tensor=True)
            # only consider items as features
            x_batch = x_batch[:, 1:]
            self.optimizer.zero_grad()
            y_hat, h = self.srnn.forward(x_batch, self.h_init)
            loss = loss_mse(y_true=np.transpose(y_batch), y_hat=y_hat)



        return loss


class SRNN(nn.Module):

    def __init__(self, n_items, h_dim_size, gru_hidden_size, n_layers=3, use_cuda=False, batch_size=32, use_logit=False):
        super(SRNN, self).__init__()
        self.batch_size = batch_size
        self.n_items = n_items
        self.h_dim_size = h_dim_size
        self.gru_hidden_size = gru_hidden_size
        self.n_layers = n_layers
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.use_cuda = use_cuda
        self.gru = nn.GRU(input_size=self.h_dim_size, hidden_size=self.h_dim_size, num_layers=self.n_layers)
        self.activation = nn.Tanh()
        self.out = nn.Linear(h_dim_size, 1)
        self.embedding = EmbeddingGrad(n_items, h_dim_size, use_cuda=use_cuda)
        self.use_logit = use_logit
        self.logistic = torch.nn.Sigmoid()
        if use_cuda:
            self = self.cuda()


    def forward(self, users, items):
        embedded = self.embedding(items)
        #embedded = embedded.unsqueeze(0)
        o, h = self.gru(torch.transpose(torch.squeeze(embedded), 0, 1))
        # o = o.view(-1, o.size(-1))

        y_hat = torch.squeeze(self.activation(self.out(o)))

        if self.use_logit:
            y_hat = self.logistic(y_hat)
        return torch.transpose(y_hat, 0, 1)


    def one_hot(self, input):
        self.one_hot_embedding.zero_()
        index = input.view(-1, 1)
        one_hot = self.one_hot_embedding.scatter_(1, index, 1)
        return one_hot

    def init_onehot_embedding(self):
        onehot = torch.FloatTensor(self.batch_size, self.n_items)
        onehot = onehot.to(self.device)
        return onehot

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

    def predict(self, X_test):

        n_samples = X_test.shape[0]
        h = torch.zeros(self.n_layers, n_samples, self.h_dim_size).to(self.device)

        return self.forward(X_test, hidden=h)


if __name__ == "__main__":
    params = {
        'batch_size': 32,
        'k': 5,
        'h_dim': 256,
        'n_epochs': 100,
        'lr': 1e-3,
        'loss_step': 1,
        'eps': 0,
        'seq_len': 4
    }

    df = pd.read_csv(cfg.vals['movielens_dir'] + "/preprocessed/ratings.csv")
    data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"
    stats = load_dict_output(data_dir, "stats.json")
    user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
    item_rating_map = load_dict_output(data_dir, "item_rating.json", True)

    interactions = Interactions(user_ids=df['user_id'].values,
                                item_ids=df['item_id'].values,
                                ratings=df['rating'].values,
                                timestamps=df['timestamp'].values,
                                num_users=stats['n_users'],
                                num_items=stats['n_items'])

    sequence_users, sequences, y, n_items = interactions.to_sequence(max_sequence_length=params['seq_len'],
                                                                     min_sequence_length=params['seq_len'])

    X = np.concatenate((sequence_users.reshape(-1, 1), sequences), axis=1)

    srnn = SRNN(stats['n_items'], h_dim_size=256, gru_hidden_size=32, n_layers=1)
    trainer = SRNNTrainer(srnn, [X, y], params, use_utility_loss=True, user_item_rating_map=user_item_rating_map,
                          item_rating_map=item_rating_map, k=5, use_cuda=False)
    trainer.train()
