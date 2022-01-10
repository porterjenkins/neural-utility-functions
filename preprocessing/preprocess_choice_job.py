import argparse

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.utils import preprocess_user_item_choice_df, write_dict_output, get_amazon_datasets
import pandas as pd
import config.config as cfg
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument("--nrows", type = int, help="limit number of rows")
parser.add_argument("--dataset", type = str, help = "dataset to process: {amazon, movielens}")
parser.add_argument("--test_user_size", type=int, help = "the number of items to sample per user", default=50)
args = parser.parse_args()


if args.dataset == "movielens":

    df = pd.read_csv(cfg.vals['movielens_dir'] + "/ratings.csv", nrows=args.nrows)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    out_dir = cfg.vals['movielens_dir'] + "/preprocessed_choice/"

elif args.dataset == "amazon":

    df = get_amazon_datasets(cfg.vals['amazon_dir'])

    df.rename(columns={"overall": "rating",
               "reviewerID": "user_id",
               "asin": "item_id",
               "unixReviewTime": "timestamp"},
              inplace=True)
    df = df[['user_id', 'item_id', 'rating', 'timestamp']]
    out_dir = cfg.vals['amazon_dir'] + "/preprocessed_choice/"

else:
    raise ValueError("--dataset must be 'amazon' or 'movielens'")

# transform rating to choice

df['rating'] = 1.0

X_train, X_test, y_test, user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats = preprocess_user_item_choice_df(
    df[['user_id', 'item_id', 'rating', 'timestamp']], test_size_per_user=args.test_user_size)



write_dict_output(out_dir, "user_item_rating.json", user_item_rating_map)
write_dict_output(out_dir, "item_rating.json", item_rating_map)
write_dict_output(out_dir, "user_id_map.json", user_id_map)
write_dict_output(out_dir, "id_user_map.json", id_user_map)
write_dict_output(out_dir, "item_id_map.json", item_id_map)
write_dict_output(out_dir, "id_item_map.json", id_item_map)
write_dict_output(out_dir, "stats.json", stats)


X_train = pd.DataFrame(X_train, columns=['user_id', 'item_id', 'rating', 'timestamp'])

X_train[['user_id', 'item_id', 'timestamp']].to_csv(out_dir + "x_train.csv", index=False)
X_train[['rating']].to_csv(out_dir + "y_train.csv", index=False)

X_test = pd.DataFrame(X_test, columns=['user_id', 'item_id', 'timestamp'])
y_test = pd.DataFrame(y_test, columns=['rating'])

X_test.to_csv(out_dir + "x_test.csv", index=False)
y_test.to_csv(out_dir + "y_test.csv", index=False)