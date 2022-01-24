##
import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing.utils import split_train_test_user, load_dict_output
import numpy as np
import pdb
import math

from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
import config.config as cfg

class MovieDataSet(Dataset):

    def __init__(self, users, items, y, user_item_rating_map, item_rating_map, c_size, s_size, n_item):

        assert users.ndim > 1
        assert items.ndim > 1
        assert len(users) == len(items)

        self.users = users
        # self.items = items
        self.y = y
        self.n_samples = self.users.shape[0]
        self.n_item = n_item if n_item else items.max()
        self.user_item_rating_map = user_item_rating_map
        self.item_rating_map = item_rating_map
        self.one_hot_items, self.items = self._get_item_one_hot(items)
        self.items = self.items.A
        self.c_size = c_size
        self.s_size = s_size

    def __getitem__(self, index):

        #generate complement and supplement sets for the index

        items = self.items[index,:].reshape(1,-1)
        users = self.users[index, :]
        y_batch = self.y[index,:]

        X_c, y_c = self.get_complement_set(users, items)
        X_s, y_s =  self.get_supp_set(users, items)

        batch = {'users': users,
                 'items': items,
                 'y':     y_batch,
                 'x_c':   X_c,
                 'y_c':   y_c,
                 'x_s':   X_s,
                 'y_s':   y_s
                 }

        return batch

    def __len__(self): #likely unnecessary
        return self.n_samples


    def _get_item_one_hot(self, items):
        one_hot_items = OneHotEncoder(categories=[range(self.n_item)], sparse=True)
        items = one_hot_items.fit_transform(items).astype(np.float32)
        return one_hot_items, items


    def get_complement_set(self, user, items):

        X_c = np.zeros((1, self.c_size, items.shape[1]), dtype=np.float32)
        y_c = np.zeros((1, self.c_size), dtype=np.float32)

        item_ratings = self.user_item_rating_map[user[0]]
        item_sampled = np.random.choice(list(item_ratings.keys()), size=self.c_size, replace=True)

        for j, item in enumerate(item_sampled):
            X_c[0, j, int(item)] = 1
            y_c[0, j] = item_ratings[item]

        return X_c, y_c


    def get_supp_set(self, user, items):

        X_s = np.zeros((1, self.s_size, items.shape[1]), dtype=np.float32)
        y_s = np.zeros((1, self.s_size), dtype=np.float32)


        user_items = list(self.user_item_rating_map[user[0]].keys())

        supp_cntr = 0
        s_set = np.zeros(self.s_size, dtype=np.int64)

        while supp_cntr < self.s_size:
            item = np.random.randint(0, self.n_item, 1)[0]
            if item not in user_items:
                s_set[supp_cntr] = item

                # handle case where item appears in test data, but not training data
                try:
                    item_ratings = self.item_rating_map[item]
                except KeyError:
                    continue


                n_ratings = len(item_ratings)
                ratings_idx = np.random.randint(0, n_ratings, 1)[0]


                X_s[0, supp_cntr, item] = 1
                y_s[0, supp_cntr] = item_ratings[ratings_idx]

                supp_cntr +=1


        return X_s, y_s



if __name__ == "__main__":
    batch_size = 8
    num_epochs = 10
    num_workers = 3

    seed = 0
    np.random.seed(seed)  # For testing/comparing results

    initialTime = time.time()
    data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"

    df = pd.read_csv(data_dir + "ratings.csv")

    X = df[['user_id', 'item_id']].to_numpy()
    y = df['rating'].to_numpy().reshape(-1, 1)

    user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
    item_rating_map = load_dict_output(data_dir, "item_rating.json", True)
    stats = load_dict_output(data_dir, "stats.json")

    X_train, X_test, y_train, y_test = split_train_test_user(X, y, random_seed=seed)

    dataset = MovieDataSet(users=np.array(X_train[:,0].reshape(-1,1)), items=np.array(X_train[:,1].reshape(-1,1)), y=y_train.reshape(-1, 1),
                            user_item_rating_map=user_item_rating_map, item_rating_map=item_rating_map, c_size=5, s_size=5, n_item=stats['n_items'])

    # num_workers denotes the amount of subprocesses you want.

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #dummy training loop

    total_samples = len(dataset)
    num_iterations = math.ceil(total_samples/ batch_size)
    a = 0

    for epoch in range(num_epochs):
        for batch in enumerate(dataloader):
            print(batch)
            quit()


    print(f'Elapsed Time: {time.time() - initialTime}')
