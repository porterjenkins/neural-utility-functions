import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.utils import preprocess_user_item_df
import numpy as np
import pandas as pd
from model.encoder import UtilityEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import math
from model.trainer import NeuralUtilityTrainer
from model._loss import loss_mse
import torch
from experiments.utils import compute_pariwise_mrs, mrs_error, get_analytical_ces_mrs, get_analytical_cobb_douglas_mrs
from experiments.utils import get_mrs_mat, logit, cobb_douglas, ces

from sklearn.preprocessing import OneHotEncoder

UTILITY = sys.argv[1]

print("Running simulation with {} utility".format(UTILITY))


RANDOM_SEED = 1990
N_USERS = 100
N = 1024
N_SIM = 10
RHO = 2

assert UTILITY in ['cobb-douglas', 'ces']

np.random.seed(RANDOM_SEED)

#weights = np.transpose(np.random.multivariate_normal(np.zeros(N), np.eye(N), 1)).flatten()
weights = np.random.uniform(0, 10, N)
weights = weights / np.sum(weights)




def gen_bundle(n, k):
    idx = np.random.randint(0, n, k)
    x = np.zeros(n)

    for i in idx:
        x[i] = 1.0

    return x, idx


def gen_dataset(n_items, n_users, n_reviews, utility_type):

    n_samples = n_users * n_reviews

    X = np.zeros((n_samples, 2), dtype=np.int32)
    y = np.zeros((n_samples, 1), dtype=np.float32)

    cntr = 0
    for i in range(n_users):
        items_reviews_i = np.random.permutation(n_items)[:n_reviews]
        for j in items_reviews_i:

            x_vec, x_i = gen_bundle(N, 1)

            if utility_type == 'cobb-douglas':
                u = cobb_douglas(x_vec, weights)
            elif utility_type == 'ces':
                u = ces(x_vec, weights, RHO)

            X[cntr, 0] = i


            X[cntr, 1] = j
            y[cntr] = u

            cntr += 1

    return X, y


def clip(arr, k):

    n = len(arr)
    arr.sort()
    return arr[:n-k]



X, y = gen_dataset(N, N_USERS, N // 2, utility_type=UTILITY)

print(X.shape)

if UTILITY == 'cobb-douglas':
    MRS = get_mrs_mat(x=np.arange(N), w=weights, mrs_func=get_analytical_cobb_douglas_mrs)
else:
    MRS = get_mrs_mat(x=np.arange(N), w=weights, mrs_func=get_analytical_ces_mrs, rho=RHO)

df = pd.DataFrame(np.concatenate([X, y], axis=1), columns=['user_id', 'item_id', 'rating'])
df, user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats = preprocess_user_item_df(df)

#X = df[['user_id', 'item_id']].astype(np.int64)
X = X.astype(np.int64)
#y = df['rating']
#X_train, X_test, y_train, y_test = split_train_test_user(X, y, random_seed=1990)




### Train Models

batch_size = 32
k = 5
h_dim = 16
n_epochs = 10
lr = 5e-5
loss_step = 10
eps = 0.01

output = {"linear": [],
          "vanilla": [],
          "utility": []}

params = {
    "h_dim_size": 256,
    "n_epochs": 20,
    "batch_size": 32,
    "lr": 1e-5,
    "eps": 1e-6,
    "c_size": 5,
    "s_size": 5,
    "loss_step": 20,
    "eval_k": 5,
    "loss": "utility",
    "lambda": .1
}

one_hot = OneHotEncoder(categories=[range(N)])
items_for_grad = one_hot.fit_transform(np.arange(N).reshape(-1,1)).todense().astype(np.float32)



df = pd.DataFrame(np.concatenate((X, y), axis=1), columns = ['users', 'items', 'rating'])
item_means = df[['items', 'rating']].groupby("items").mean()

users = torch.from_numpy(np.arange(N).reshape(-1,1))
items_for_grad = torch.from_numpy(items_for_grad)
y_true = torch.from_numpy(item_means.values.reshape(-1,1))

for iter in range(N_SIM):

    # Train Linear Regression
    enc = OneHotEncoder(sparse=False)
    X_train_sparse = enc.fit_transform(X=X[:, 1].reshape(-1,1))

    linear = LinearRegression(fit_intercept=False)
    linear.fit(X_train_sparse, y)



    grad_linear = linear.coef_.flatten()
    mrs_linear = compute_pariwise_mrs(grad_linear)
    l2_linear = mrs_error(MRS, mrs_linear)
    output['linear'].append(l2_linear)


    # Train Vanilla Wide&Deep
    encoder = UtilityEncoder(stats['n_items'], h_dim_size=params["h_dim_size"], use_cuda=False)

    trainer = NeuralUtilityTrainer(users=X[:, 0].reshape(-1, 1), items=X[:, 1].reshape(-1, 1),
                                   y_train=y, model=encoder, loss=loss_mse,
                                   n_epochs=params['n_epochs'], batch_size=params['batch_size'],
                                   lr=params["lr"], loss_step_print=params["loss_step"],
                                   eps=params["eps"], item_rating_map=item_rating_map,
                                   user_item_rating_map=user_item_rating_map,
                                   c_size=params["c_size"], s_size=params["s_size"],
                                   n_items=stats["n_items"], use_cuda=False,
                                   model_name=None, model_path=None,
                                   checkpoint=False, lmbda=params["lambda"])

    _ = trainer.fit()

    grad_vanilla = NeuralUtilityTrainer.get_gradient(encoder, loss_mse, users, items_for_grad, y_true)
    mrs_vanilla = compute_pariwise_mrs(grad_vanilla)
    l2_vanilla = mrs_error(MRS, mrs_vanilla)
    output['vanilla'].append(l2_vanilla)



    # Train Neural Utility Function


    encoder_utility = UtilityEncoder(stats['n_items'], h_dim_size=params["h_dim_size"], use_cuda=False)

    print("Model intialized")
    print("Beginning Training...")

    trainer = NeuralUtilityTrainer(users=X[:, 0].reshape(-1,1), items=X[:, 1].reshape(-1,1),
                                   y_train=y, model=encoder_utility, loss=loss_mse,
                                   n_epochs=params['n_epochs'], batch_size=params['batch_size'],
                                   lr=params["lr"], loss_step_print=params["loss_step"],
                                   eps=params["eps"], item_rating_map=item_rating_map,
                                   user_item_rating_map=user_item_rating_map,
                                   c_size=params["c_size"], s_size=params["s_size"],
                                   n_items=stats["n_items"], use_cuda=False,
                                   model_name=None, model_path=None,
                                   checkpoint=False, lmbda=params["lambda"])

    _ = trainer.fit_utility_loss()



    grad_utility =  NeuralUtilityTrainer.get_gradient(encoder_utility, loss_mse, users, items_for_grad, y_true)
    mrs_utility = compute_pariwise_mrs(grad_utility)

    l2_utility = mrs_error(MRS, mrs_utility)
    output['utility'].append(l2_utility)



for k, v in output.items():

    print(k)


    v_clip = clip(v, k=2)

    print(v_clip)
    print("mean: {:.4f}".format(np.mean(v_clip)))
    print("std: {:.4f}".format(np.std(v_clip)))