import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import config.config as cfg
from preprocessing.utils import split_train_test_user, load_dict_output
from preprocessing.interactions import Interactions
import numpy as np
from baselines.s_rnn import SRNN, SRNNTrainer
from experiments.utils import get_eval_metrics_sequential
import argparse
from experiments.utils import get_test_sample_size, read_train_test_dir, log_output
from model.trainer import SequenceTrainer
from model._loss import loss_mse


parser = argparse.ArgumentParser()
parser.add_argument("--loss", type = str, help="loss function to optimize", default='mse')
parser.add_argument("--cuda", type = bool, help="flag to run on gpu", default=False)
parser.add_argument("--checkpoint", type = bool, help="flag to run on gpu", default=True)
parser.add_argument("--dataset", type = str, help = "dataset to process: {amazon, movielens}", default="Movielens")
parser.add_argument("--epochs", type = int, help = "Maximum number of epochs for training", default=1)
parser.add_argument("--eps", type = float, help = "Tolerance for early stopping", default=1e-3)
parser.add_argument("--h_dim_size", type = int, help = "Size of embedding dimension", default=256)
parser.add_argument("--batch_size", type = int, help = "Size of training batch", default=32)
parser.add_argument("--lr", type = float, help = "Learning Rate", default=5e-5)
parser.add_argument("--c_size", type = int, help = "Size of complement set", default=5)
parser.add_argument("--s_size", type = int, help = "Size of supplement set", default=5)
parser.add_argument("--lmbda", type = float, help = "Size of supplement set", default=.1)
parser.add_argument("--seq_len", type = int, help = "Length of sequences", default=4)
parser.add_argument("--max_iter", type = int, help = "Length of sequences", default=None)
parser.add_argument("--grad_clip", type=float, help = "Clip gradients during trining", default=None)




args = parser.parse_args()

MODEL_NAME = "srnn_ratings_{}_{}".format(args.dataset, args.loss)
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
            "seq_len": args.seq_len,
            "max_iter": args.max_iter,
            "grad_clip": args.grad_clip
        }


print("Reading dataset")

if args.dataset == "movielens":
    data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"
elif args.dataset == "amazon":
    data_dir = cfg.vals['amazon_dir'] + "/preprocessed/"
else:
    raise ValueError("--dataset must be 'amazon' or 'movielens'")



X_train, X_test, y_train, y_test = read_train_test_dir(data_dir, drop_ts=False)

user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
item_rating_map = load_dict_output(data_dir, "item_rating.json", True)
stats = load_dict_output(data_dir, "stats.json")
print("Dataset read complete...")

print("n users: {}".format(stats['n_users']))
print("n items: {}".format(stats['n_items']))



interactions = Interactions(user_ids=X_train[:, 0],
                            item_ids=X_train[:, 1],
                            ratings=y_train.flatten(),
                            timestamps=X_train[:, 2],
                            num_users=stats['n_users'],
                            num_items=stats['n_items'])

sequence_users, sequences, y_seq, n_items = interactions.to_sequence(max_sequence_length=params['seq_len'],
                                                                 min_sequence_length=params['seq_len'])

X = np.concatenate((sequence_users.reshape(-1, 1), sequences), axis=1)
y_seq = y_seq.astype(np.float32)

model = SRNN(stats['n_items'], h_dim_size=256, gru_hidden_size=32, n_layers=1)

print("Model intialized")
print("Beginning Training...")

trainer = SequenceTrainer(users=sequence_users.reshape(-1,1), items=sequences,
                          y_train=y_seq, model=model, loss=loss_mse,
                          n_epochs=params['n_epochs'], batch_size=params['batch_size'],
                          lr=params["lr"], loss_step_print=params["loss_step"],
                          eps=params["eps"], item_rating_map=item_rating_map,
                          user_item_rating_map=user_item_rating_map,
                          c_size=params["c_size"], s_size=params["s_size"],
                          n_items=stats["n_items"], use_cuda=args.cuda,
                          model_name=MODEL_NAME, model_path=MODEL_DIR,
                          checkpoint=args.checkpoint, lmbda=params["lambda"],
                          seq_len=params["seq_len"], parallel=False,
                          max_iter=params["max_iter"], grad_clip=params["grad_clip"])


if params['loss'] == 'utility':
    print("utility loss")
    trainer.fit_utility_loss()
else:
    print("mse loss")
    trainer.fit()





interactions = Interactions(user_ids=X_test[:, 0],
                            item_ids=X_test[:, 1],
                            ratings=y_test.flatten(),
                            timestamps=X_test[:, 2],
                            num_users=stats['n_users'],
                            num_items=stats['n_items'])

users_test, items_test, y_test_seq, _ = interactions.to_sequence(max_sequence_length=params['seq_len'],
                                                                 min_sequence_length=params['seq_len'])

n_test = get_test_sample_size(users_test.shape[0], k=TEST_BATCH_SIZE)
users_test = users_test[:n_test].reshape(-1,1)
items_test = items_test[:n_test, :]
y_test_seq = y_test_seq[:n_test, :]

preds = trainer.predict(users=users_test, items=items_test, y=y_test_seq,
                        batch_size=TEST_BATCH_SIZE)

output, rmse, dcg = get_eval_metrics_sequential(users_test, preds, y_test_seq, params["seq_len"], params["eval_k"])

print("rmse: {:.4f}".format(rmse))
print("dcg: {:.4f}".format(dcg))

log_output(MODEL_DIR, MODEL_NAME, params, output=[rmse, dcg])