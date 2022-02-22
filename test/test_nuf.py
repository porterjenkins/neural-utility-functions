import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import pandas as pd
from preprocessing.utils import split_train_test_user, load_dict_output
from model.trainer import NeuralUtilityTrainer
from model.neural_utility_function import NeuralUtility
import numpy as np
from model._loss import loss_mse
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_size, h_1_size, h_2_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, h_1_size)
        self.fc2 = nn.Linear(h_1_size, h_2_size)
        self.output = nn.Linear(h_2_size, 1)

    def forward(self, x):

        h = self.fc1(x)
        h = self.fc2(h)
        y_hat = self.output(h)

        return y_hat





data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"

df = pd.read_csv(data_dir + "ratings.csv")

X = df[['user_id', 'item_id']].values.astype(np.int64)
y = df['rating'].values.reshape(-1, 1)

user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
item_rating_map = load_dict_output(data_dir, "item_rating.json", True)
stats = load_dict_output(data_dir, "stats.json")

X_train, X_test, y_train, y_test = split_train_test_user(X, y)


mlp = MLP(64, 32, 16)

model = NeuralUtility(backbone=mlp, n_items=stats['n_items'], h_dim_size=64)


trainer = NeuralUtilityTrainer(X_train=X_train, y_train=y_train, model=model, loss=loss_mse, \
                               n_epochs=5, batch_size=32, lr=1e-3, loss_step_print=25, eps=.001,
                               item_rating_map=item_rating_map, user_item_rating_map=user_item_rating_map,
                               c_size=5, s_size=5, n_items=stats["n_items"], checkpoint=True,
                               model_path=None, model_name='test_nuf')

#trainer.fit()
trainer.fit_utility_loss()