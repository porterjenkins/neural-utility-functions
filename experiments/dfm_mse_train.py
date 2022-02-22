import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
from preprocessing.utils import load_dict_output
from model.trainer import NeuralUtilityTrainer
import numpy as np
from model._loss import loss_mse
from baselines.deep_fm import DeepFM
import torch
from experiments.utils import get_eval_metrics, log_output
import argparse
import pandas as pd
from experiments.utils import get_test_sample_size, read_train_test_dir



parser = argparse.ArgumentParser()
parser.add_argument("--loss", type = str, help="loss function to optimize", default='mse')
parser.add_argument("--cuda", type = bool, help="flag to run on gpu", default=False)
parser.add_argument("--checkpoint", type = bool, help="flag to run on gpu", default=True)
parser.add_argument("--dataset", type = str, help = "dataset to process: {amazon, movielens}", default="movielens")
parser.add_argument("--epochs", type = int, help = "Maximum number of epochs for training", default=1)
parser.add_argument("--eps", type = float, help = "Tolerance for early stopping", default=1e-3)
parser.add_argument("--h_dim_size", type = int, help = "Size of embedding dimension", default=256)
parser.add_argument("--batch_size", type = int, help = "Size of training batch", default=32)
parser.add_argument("--lr", type = float, help = "Learning Rate", default=5e-5)
parser.add_argument("--c_size", type = int, help = "Size of complement set", default=5)
parser.add_argument("--s_size", type = int, help = "Size of supplement set", default=5)
parser.add_argument("--lmbda", type = float, help = "Size of supplement set", default=.1)
parser.add_argument("--max_iter", type = int, help = "Length of sequences", default=None)





args = parser.parse_args()

MODEL_NAME = "dfm_ratings_{}_{}".format(args.dataset, args.loss)
MODEL_DIR = cfg.vals['model_dir']
TEST_BATCH_SIZE = 100
RANDOM_SEED = 1990
LOSS_STEP = 50
EVAL_K = 5

params = {
            "h_dim_size": args.h_dim_size,
            "n_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "eps": args.eps,
            "c_size": args.c_size,
            "s_size": args.s_size,
            "loss_step": LOSS_STEP,
            "eval_k": EVAL_K,
            "loss": args.loss,
            "lambda": args.lmbda,
            "max_iter": args.max_iter
        }


print("Reading dataset")

if args.dataset == "movielens":
    data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"
elif args.dataset == "amazon":
    data_dir = cfg.vals['amazon_dir'] + "/preprocessed/"
else:
    raise ValueError("--dataset must be 'amazon' or 'movielens'")



X_train, X_test, y_train, y_test = read_train_test_dir(data_dir)
print("Dataset read complete...")


user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
item_rating_map = load_dict_output(data_dir, "item_rating.json", True)
stats = load_dict_output(data_dir, "stats.json")

print("n users: {}".format(stats['n_users']))
print("n items: {}".format(stats['n_items']))


n_test = get_test_sample_size(X_test.shape[0], k=TEST_BATCH_SIZE)
X_test = X_test[:n_test, :]
y_test = y_test[:n_test, :]

model = DeepFM(field_dims=[stats["n_items"]], embed_dim=params["h_dim_size"], mlp_dims=(16, 16), dropout=0.2)


print("Model intialized")
print("Beginning Training...")

trainer = NeuralUtilityTrainer(users=X_train[:, 0].reshape(-1,1), items=X_train[:, 1:].reshape(-1,1),
                               y_train=y_train, model=model, loss=loss_mse,
                               n_epochs=params['n_epochs'], batch_size=params['batch_size'],
                               lr=params["lr"], loss_step_print=params["loss_step"],
                               eps=params["eps"], item_rating_map=item_rating_map,
                               user_item_rating_map=user_item_rating_map,
                               c_size=params["c_size"], s_size=params["s_size"],
                               n_items=stats["n_items"], use_cuda=args.cuda,
                               model_name=MODEL_NAME, model_path=MODEL_DIR,
                               checkpoint=args.checkpoint, lmbda=params["lambda"],
                               max_iter=params["max_iter"])


if params['loss'] == 'utility':
    print("utility loss")
    trainer.fit_utility_loss()
else:
    print("mse loss")
    trainer.fit()


users_test = X_test[:, 0].reshape(-1,1)
items_test = X_test[:, 1].reshape(-1,1)
y_test = y_test.reshape(-1,1)


preds = trainer.predict(users=users_test, items=items_test, y=y_test,
                        batch_size=TEST_BATCH_SIZE).reshape(-1,1)


output = pd.DataFrame(np.concatenate((users_test, preds, y_test), axis=1),
                      columns = ['user_id', 'pred', 'y_true'])

output, rmse, dcg = get_eval_metrics(output, at_k=params['eval_k'])

print("rmse: {:.4f}".format(rmse))
print("dcg: {:.4f}".format(dcg))

log_output(MODEL_DIR, MODEL_NAME, params, output=[rmse, dcg])