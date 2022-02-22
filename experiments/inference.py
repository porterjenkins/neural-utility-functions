import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import torch
import argparse
from experiments.utils import get_test_sample_size, read_train_test_dir
from experiments.utils import get_eval_metrics, get_choice_eval_metrics
from model.predictor import Predictor
import pandas as pd
import numpy as np
from preprocessing.utils import load_dict_output


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type = bool, help="flag to run on gpu", default=False)
parser.add_argument("--dataset", type = str, help = "dataset to process: {amazon, movielens}", default="Movielens")
parser.add_argument("--task", type=str, help = "'choice' or 'ratings'")

args = parser.parse_args()

assert args.task in ["choice", "ratings"]

TEST_BATCH_SIZE = 100
RANDOM_SEED = 1990
EVAL_K = 5

model_path = cfg.vals["model_dir"] + "/wide_deep_amazon_logit_done.pt"



print("Reading dataset")

if args.dataset == "movielens":
    data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"
elif args.dataset == "amazon":
    data_dir = cfg.vals['amazon_dir'] + "/preprocessed/"
else:
    raise ValueError("--dataset must be 'amazon' or 'movielens'")



X_train, X_test, y_train, y_test = read_train_test_dir(data_dir)
stats = load_dict_output(data_dir, "stats.json")
print("Dataset read complete...")



n_test = get_test_sample_size(X_test.shape[0], k=TEST_BATCH_SIZE)
X_test = X_test[:n_test, :]
y_test = y_test[:n_test, :]

users_test = X_test[:, 0].reshape(-1,1)
items_test = X_test[:, 1].reshape(-1,1)
y_test = y_test.reshape(-1,1)


predictor = Predictor(model=model, batch_size=TEST_BATCH_SIZE, users=users_test, items=items_test, y=y_test,
                      use_cuda=args.cuda, n_items=stats["n_items"])



preds = predictor.predict().reshape(-1,1)


output = pd.DataFrame(np.concatenate((users_test, preds, y_test), axis=1),
                      columns = ['user_id', 'pred', 'y_true'])


if args.task == "choice":

    output, hit_ratio, ndcg = get_choice_eval_metrics(output, at_k=EVAL_K)

    print("hit ratio: {:.4f}".format(hit_ratio))
    print("ndcg: {:.4f}".format(ndcg))

else:

    output, rmse, dcg = get_eval_metrics(output, at_k=EVAL_K)

    print("rmse: {:.4f}".format(rmse))
    print("dcg: {:.4f}".format(dcg))

