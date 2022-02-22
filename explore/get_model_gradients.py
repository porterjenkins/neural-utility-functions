import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
from experiments.utils import read_train_test_dir
from preprocessing.utils import load_dict_output
import torch
import pandas as pd
import numpy as np
from model.trainer import NeuralUtilityTrainer
from model._loss import loss_mse
from sklearn.preprocessing import OneHotEncoder
from experiments.utils import get_mrs_arr, get_supp_k, get_comp_k
from preprocessing.utils import load_dict_output

# Set model and data paths manuall
# TODO: Ahmad to modify for his application
MODEL_PATH = cfg.vals["model_dir"] + "/home_kitchen_ahmad/item_encoder_amazon_utility_done.pt"
data_dir = cfg.vals['amazon_dir'] + "/preprocessed_home_kitchen/"
item_map = load_dict_output(cfg.vals["amazon_dir"] + "/preprocessed_home_kitchen/" , "id_item_map.json", True)
item_to_idx = load_dict_output(cfg.vals["amazon_dir"] + "/preprocessed_home_kitchen/" , "item_id_map.json", True)


# Process data
X_train, X_test, y_train, y_test = read_train_test_dir(data_dir)
stats = load_dict_output(data_dir, "stats.json")

one_hot = OneHotEncoder(categories=[range(stats["n_items"])])
items_for_grad = one_hot.fit_transform(np.arange(stats["n_items"]).reshape(-1,1)).todense().astype(np.float32)

train = np.concatenate([X_train, y_train],axis=1)
test = np.concatenate([X_test, y_test],axis=1)

X = np.concatenate((train, test), axis=0)

df = pd.DataFrame(X, columns=["user_id", "item_id", "rating"])
item_means = df[['item_id', 'rating']].groupby("item_id").mean()

users=None
items_for_grad = torch.from_numpy(items_for_grad)
y_true = torch.from_numpy(item_means.values.reshape(-1,1))


# Load Model
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))


# compute gradietns
gradients =  NeuralUtilityTrainer.get_gradient(model, loss_mse, users, items_for_grad, y_true)
print(gradients)


mrs = get_mrs_arr(gradients)
item_idx = item_to_idx[1]



item = mrs[item_idx, :]
print(item)
supp = get_supp_k(item, k=5)
comp = get_comp_k(item, k=5)

print("Item: {}".format(item_map[item_idx]))
print("Supplements: ")
for i in supp:
    print(item_map[i], i, gradients[i])
print("Complements: ")
for i in comp:
    print(item_map[i], i, gradients[i])



