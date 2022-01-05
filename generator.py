import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import pandas as pd
import numpy as np
from preprocessing.utils import split_train_test_user, load_dict_output
import torch
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix

class Generator(object): #Todo: update to use pytorch dataloaders/datasets.

    def __init__(self, users, items, y, batch_size, shuffle, n_item):

        assert users.ndim > 1
        assert items.ndim > 1
        assert len(users) == len(items)


        self.users = users
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = self.users.shape[0]
        self.curr_idx = 0
        self.epoch_cntr = 0
        self.n_item = n_item if n_item else items.max()
        self.one_hot_items, self.items = self._get_item_one_hot(items)
        self.idx = self._get_index()


    def _get_item_one_hot(self, items):
        one_hot_items = OneHotEncoder(categories=[range(self.n_item)], sparse=True)
        items = one_hot_items.fit_transform(items).astype(np.float32)
        return one_hot_items, items

    def update_data(self, users, items, y, shuffle, batch_size):
        self.users = users
        self.items = self.one_hot_items.fit_transform(items).astype(np.float32)
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_samples = self.users.shape[0]
        self.idx = self._get_index()
        self.curr_idx = 0
        self.epoch_cntr = 0

    def _get_index(self):
        if self.shuffle:
            return np.random.permutation(np.arange(self.n_samples))
        else:
            return np.arange(self.n_samples)



    def shuffle_idx(self):
        self.idx = np.random.permutation(np.arange(self.n_samples))

    def reset(self):
        print("End of epoch: {}".format(self.epoch_cntr))
        self.curr_idx = 0
        self.epoch_cntr += 1
        if self.shuffle:
            self.shuffle_idx()

    def check(self):
        if self.curr_idx + self.batch_size >= self.n_samples:
            return True
        else:
            return False


    def update_curr_idx(self):
        self.curr_idx += self.batch_size

    def reset_epoch(self):
        self.epoch_cntr = 0

    def get_sparse_tensor(self, sparse_mtx):

        coo = coo_matrix(sparse_mtx)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def get_batch(self, as_tensor):
        reset = self.check()
        if reset:
            self.reset()

        batch_idx = self.idx[self.curr_idx:(self.curr_idx + self.batch_size)]
        items = self.items[batch_idx, :]

        import numpy as np
        if items.dtype == np.float64:
            stop = 0

        users = self.users[batch_idx, :]
        y_batch = self.y[batch_idx, :]
        self.update_curr_idx()

        if as_tensor:
            items = torch.from_numpy(items.todense())
            #items = self.get_sparse_tensor(items)
            users = torch.from_numpy(users)
            y_batch = torch.from_numpy(y_batch)

        batch = {'users': users,
                 'items': items,
                 'y': y_batch}

        return batch



class CoocurrenceGenerator(Generator): #Todo: update to use pytorch dataloaders/datasets.
    """
    Class to generate minibatch samples, as well as complement and supplement sets
        - Assume that first column of X is user_id, second column is item_id
    """


    def __init__(self, users, items, y, batch_size, shuffle, user_item_rating_map, item_rating_map, c_size, s_size, n_item):
        super().__init__(users, items, y, batch_size, shuffle, n_item)
        self.user_item_rating_map = user_item_rating_map
        self.item_rating_map = item_rating_map
        self.c_size = c_size
        self.s_size = s_size



    def get_complement_set(self, users, items):
        X_c = np.zeros((items.shape[0], self.c_size, items.shape[1]), dtype=np.float32)
        y_c = np.zeros((users.shape[0], self.c_size), dtype=np.float32)


        for i, user_id in enumerate(users):
            item_ratings = self.user_item_rating_map[user_id[0]]
            item_sampled = np.random.choice(list(item_ratings.keys()), size=self.c_size, replace=True)


            for j, item in enumerate(item_sampled):
                X_c[i, j, int(item)] = 1
                y_c[i, j] = item_ratings[item]

        return X_c, y_c

    def get_supp_set(self, users, items):

        X_s = np.zeros((items.shape[0], self.s_size, items.shape[1]), dtype=np.float32)
        y_s = np.zeros((users.shape[0], self.s_size), dtype=np.float32)


        for i, user_id in enumerate(users):
            user_items = list(self.user_item_rating_map[user_id[0]].keys())

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


                    X_s[i, supp_cntr, item] = 1
                    y_s[i, supp_cntr] = item_ratings[ratings_idx]

                    supp_cntr +=1


        return X_s, y_s


    def get_batch(self, as_tensor=False):

        b = super(CoocurrenceGenerator, self).get_batch(False)

        X_c, y_c = self.get_complement_set(b['users'], b['items'])
        X_s, y_s = self.get_supp_set(b['users'], b['items'])

        if as_tensor:
            b['items'] = torch.from_numpy(b['items'].todense())
            b['users'] = torch.from_numpy(b['users'])
            b['y'] = torch.from_numpy(b['y'])
            X_c = torch.from_numpy(X_c)
            X_s = torch.from_numpy(X_s)
            y_c = torch.from_numpy(y_c)
            y_s = torch.from_numpy(y_s)

        batch = {'users': b['users'],
                 'items': b['items'],
                 'y': b['y'],
                 'x_c': X_c,
                 'y_c': y_c,
                 'x_s': X_s,
                 'y_s': y_s}

        return batch

class SeqCoocurrenceGenerator(CoocurrenceGenerator): #deprecated?

    def __init__(self, users, items, y, batch_size, shuffle, user_item_rating_map, item_rating_map, c_size, s_size,
                 n_item, seq_len):
        self.seq_len = seq_len
        super().__init__(users, items, y, batch_size, shuffle, user_item_rating_map, item_rating_map, c_size, s_size, n_item)



    def _get_item_one_hot(self, items):
        categories = [range(self.n_item)]*self.seq_len
        one_hot_items = OneHotEncoder(categories=categories, sparse=True)
        items = one_hot_items.fit_transform(items).astype(np.float32)
        return one_hot_items, items

    def get_complement_set(self, users, items):

        X_c = np.zeros((users.shape[0], self.seq_len, self.c_size, self.n_item), dtype=np.float32)
        y_c = np.zeros((users.shape[0], self.seq_len, self.c_size), dtype=np.float32)


        for i, user_id in enumerate(users):
            item_ratings = self.user_item_rating_map[user_id[0]]


            for ts in range(self.seq_len):

                items_sampled = np.random.choice(list(item_ratings.keys()), size=self.c_size, replace=True)


                for j, item in enumerate(items_sampled):
                    X_c[i, ts, j, int(item)] = 1.0
                    y_c[i, ts, j] = item_ratings[item]

        return X_c, y_c


    def get_supp_set(self, users, items):

        X_s = np.zeros((users.shape[0], self.seq_len, self.c_size, self.n_item), dtype=np.float32)
        y_s = np.zeros((users.shape[0], self.seq_len, self.c_size), dtype=np.float32)


        for i, user_id in enumerate(users):
            user_items = list(self.user_item_rating_map[user_id[0]].keys())

            for ts in range(self.seq_len):

                supp_cntr = 0
                s_set = np.zeros(self.s_size, dtype=np.int64)
                y_s_set = np.zeros(self.s_size, dtype=np.float32)

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
                        y_s_set[supp_cntr] = self.item_rating_map[item][ratings_idx]

                        X_s[i, ts, supp_cntr, item] = 1
                        y_s[i, ts, supp_cntr] = item_ratings[ratings_idx]

                        supp_cntr += 1


                #X_s[i, ts, :] = s_set
                #y_s[i, ts, :] = y_s_set

        return X_s, y_s

    def get_batch(self, as_tensor=False):

        b = super(CoocurrenceGenerator, self).get_batch(False)

        b["items"] = np.array(b["items"].todense()).reshape(self.batch_size, self.seq_len, self.n_item)

        X_c, y_c = self.get_complement_set(b['users'], b['items'])
        X_s, y_s = self.get_supp_set(b['users'], b['items'])

        if as_tensor:
            b['items'] = torch.from_numpy(b['items'])
            b['users'] = torch.from_numpy(b['users'])
            b['y'] = torch.from_numpy(b['y'])
            X_c = torch.from_numpy(X_c)
            X_s = torch.from_numpy(X_s)
            y_c = torch.from_numpy(y_c)
            y_s = torch.from_numpy(y_s)

        batch = {'users': b['users'],
                 'items': b['items'],
                 'y': b['y'],
                 'x_c': X_c,
                 'y_c': y_c,
                 'x_s': X_s,
                 'y_s': y_s}

        return batch








if __name__ == "__main__":

    data_dir = cfg.vals['movielens_dir'] + "/preprocessed/"

    df = pd.read_csv(data_dir + "ratings.csv")

    X = df[['user_id', 'item_id']]
    y = df['rating']

    user_item_rating_map = load_dict_output(data_dir, "user_item_rating.json", True)
    item_rating_map = load_dict_output(data_dir, "item_rating.json", True)
    stats = load_dict_output(data_dir, "stats.json")



    X_train, X_test, y_train, y_test = split_train_test_user(X, y)

    gen = CoocurrenceGenerator(X=X_train.values, Y=y_train.values.reshape(-1,1), batch_size=8, shuffle=True,
                               user_item_rating_map=user_item_rating_map, item_rating_map=item_rating_map, c_size=5,
                               s_size=5, n_item=stats['n_items'])

    while gen.epoch_cntr < 10:

        x_batch, y_batch, X_c, y_c, X_s, y_s = gen.get_batch()
        print(x_batch)

    print(gen.epoch_cntr)
