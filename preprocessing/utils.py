import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import json
import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gzip

def get_one_hot_encodings(X, sparse=False):
    encoder = OneHotEncoder(sparse=sparse)
    if X.values.ndim == 1:
        one_hot = encoder.fit_transform(X.values.reshape(-1, 1))
    else:
        one_hot = encoder.fit_transform(X)
    cols = ['oh_{}'.format(x) for x in range(one_hot.shape[1])]
    one_hot = pd.DataFrame(one_hot, columns=cols)

    return pd.concat([X, one_hot], axis=1)



def split_train_test_user(X, y, test_size=.2, random_seed=None, strat_col=0):

    assert isinstance(X, np.ndarray)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=X[:, strat_col],
                                                        random_state=random_seed)

    return X_train, X_test, y_train, y_test


def preprocess_user_item_df(df):

    n_rows = df.shape[0]
    output = np.zeros((n_rows, 2))

    user_item_rating_map = {}
    item_rating_map = {}

    user_id_map = {}
    item_id_map = {}

    id_user_map = {}
    id_item_map = {}

    user_cntr = 0
    item_cntr = 0

    row_cntr = 0
    for idx, row in df.iterrows():

        # update values of dataframe
        if row.user_id not in user_id_map:
            user_id_map[row.user_id] = user_cntr
            id_user_map[user_cntr] = row.user_id

            user_cntr += 1

        if row.item_id not in item_id_map:
            item_id_map[row.item_id] = item_cntr
            id_item_map[item_cntr] = row.item_id

            item_cntr += 1


        # update item_rating dicts

        if item_id_map[row.item_id] not in item_rating_map:
            item_rating_map[item_id_map[row.item_id]] = []

        item_rating_map[item_id_map[row.item_id]].append(row.rating)

        # update user_item_rating dict

        if user_id_map[row.user_id] not in user_item_rating_map:
            user_item_rating_map[user_id_map[row.user_id]] = {}



        user_item_rating_map[user_id_map[row.user_id]][item_id_map[row.item_id]] = row.rating

        output[row_cntr, 0] = user_id_map[row.user_id]
        output[row_cntr, 1] = item_id_map[row.item_id]


        row_cntr += 1

        print("progress: {:.4f}%".format((row_cntr / n_rows)*100), end='\r')

    stats = {
                'n_users': user_cntr,
                'n_items': item_cntr
             }



    return output, user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats


def preprocess_user_item_choice_df(df, test_size_per_user=10):

    n_users = len(np.unique(df['user_id']))

    X_test_pos = np.zeros((n_users, 3))
    X_test_neg = np.zeros((n_users*test_size_per_user, 3))

    X_train_list = []

    n_rows = df.shape[0]


    user_item_rating_map = {}
    item_rating_map = {}

    user_id_map = {}
    item_id_map = {}

    id_user_map = {}
    id_item_map = {}

    user_cntr = 0
    item_cntr = 0

    prog_cntr = 0
    for user_id, user_data in df.groupby("user_id"):

        user_data.sort_values(by="timestamp", ascending=True, inplace=True)
        user_n = user_data.shape[0]

        user_data_train = user_data.iloc[:user_n-1, :]
        user_data_test = user_data.values[user_n-1, :]

        # add user_id to map
        if user_id not in user_id_map:
            user_id_map[user_id] = user_cntr
            id_user_map[user_cntr] = user_id
            user_cntr += 1

        # test set item to item map

        if user_data_test[1] not in item_id_map:
            item_id_map[user_data_test[1]] = item_cntr
            id_item_map[item_cntr] = user_data_test[1]

            item_cntr += 1

        # assign test sample to vector
        test_row_idx = user_id_map[user_id]

        #X_test_pos[test_row_idx, :] = user_data_test
        X_test_pos[test_row_idx, 0] = user_id_map[user_data_test[0]]
        X_test_pos[test_row_idx, 1] = item_id_map[user_data_test[1]]
        X_test_pos[test_row_idx, 2] = 0.0

        user_output = np.zeros_like(user_data_train)


        row_cntr = 0

        for idx, row in user_data_train.iterrows():

            if row.item_id not in item_id_map:
                item_id_map[row.item_id] = item_cntr
                id_item_map[item_cntr] = row.item_id

                item_cntr += 1


            # update item_rating dicts

            if item_id_map[row.item_id] not in item_rating_map:
                item_rating_map[item_id_map[row.item_id]] = []

            # set item_rating_map elements to 0.0 --> supplement set
            item_rating_map[item_id_map[row.item_id]].append(0.0)

            # update user_item_rating dict

            if user_id_map[row.user_id] not in user_item_rating_map:
                user_item_rating_map[user_id_map[row.user_id]] = {}



            user_item_rating_map[user_id_map[row.user_id]][item_id_map[row.item_id]] = row.rating

            user_output[row_cntr, 0] = user_id_map[row.user_id]
            user_output[row_cntr, 1] = item_id_map[row.item_id]
            user_output[row_cntr, 2] = row.rating
            user_output[row_cntr, 3] = row.timestamp


            row_cntr += 1
            prog_cntr += 1

            print("progress: {:.4f}%".format((prog_cntr / n_rows)*100), end='\r')

        X_train_list.append(user_output)

    stats = {
        'n_users': user_cntr,
        'n_items': item_cntr
    }

    print ("generating test set samples")

    ## Generate k samples for ranking and recommendation experiment
    test_set_neg_idx = 0

    for user, item_dict in user_item_rating_map.items():

        neg_sample_cntr = 0

        time_stamp = 1
        while neg_sample_cntr < test_size_per_user:

            neg_sample = np.random.randint(0, stats["n_items"], size=1)[0]

            if neg_sample not in set(item_dict.keys()):

                # store in test set vector
                X_test_neg[test_set_neg_idx, 0] = user
                X_test_neg[test_set_neg_idx, 1] = neg_sample
                X_test_neg[test_set_neg_idx, 2] = time_stamp


                neg_sample_cntr += 1
                test_set_neg_idx += 1
                time_stamp += 1

                print("progress: {:.4f}%".format((test_set_neg_idx / (n_users*test_size_per_user)) * 100), end='\r')


    X_test = np.concatenate((X_test_pos, X_test_neg), axis=0)

    y_test_pos = np.ones((n_users, 1))
    y_test_neg = np.zeros((n_users*test_size_per_user, 1))

    y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

    X_train = np.concatenate(X_train_list, axis=0)

    return X_train, X_test, y_test, user_item_rating_map, item_rating_map, user_id_map, id_user_map, item_id_map, id_item_map, stats



def write_dict_output(out_dir, fname, data):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_path = out_dir + fname
    with open(out_path, 'w') as fp:
        json.dump(data, fp)

def load_dict_output(dir, fname, keys_to_int=False):

    fpath = dir + fname
    with open(fpath, 'r') as fp:
        data =json.load(fp)

    if keys_to_int:
        data_to_int = {}
        for k, v in data.items():

            if isinstance(v, dict):

                try:
                    data_to_int[int(k)] = {int(k2): v2 for k2,v2 in v.items()}
                except:
                    data_to_int[int(float(k))] = {int(k2): v2 for k2, v2 in v.items()}

            else:
                try:
                    data_to_int[int(k)] = v
                except ValueError:
                    try:
                        data_to_int[int(float(k))] = v
                    except:
                        continue

        return data_to_int


    else:
        return data


true = True
false = False

def parse(path):
    g = gzip.open(path, 'rb')
    for line in g:
        yield eval(line)

def pandas_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def get_amazon_datasets(data_dir):

    data_list = list()
    for f in os.listdir(data_dir):

        if f.endswith('.gz'):
            print("getting data from {}".format(f))
            fname = "{}/{}".format(data_dir, f)
            df = pandas_df(fname)
            df = df.drop(columns=[
                "reviewerName",
                "reviewText",
                "summary",
                "reviewTime",
                "verified",
                "vote",
                "style",
                "image"
            ])
            data_list.append(df)

    data_all = pd.concat(data_list)
    return data_all



def split_train_test_sequential(df, n_items, n_users, seq_len=4, test_user_size=10):
    """
    DF must be sorted by user id and timestamp!
    Assume columns of df are in order: ['user_id', 'item_id', 'rating', 'timestamp']
    :param df:
    :return:
    """
    cols = list(df.columns) + ['seq_id']
    assert 'user_id' in cols and 'timestamp' in cols and 'item_id' in cols


    train_user_list = []
    test_user_list = []

    print("generating noise samples for sequence data")

    prog_cntr = 0
    for user_id, user_data in df.groupby('user_id'):

        if user_data.shape[0] < seq_len:
            # skip user if not enough data
            continue

        train_idx = user_data.shape[0] - 1

        train_user = user_data.iloc[:train_idx, :]

        test_user_true = user_data.iloc[(train_idx-seq_len + 1):(train_idx+1), :]
        test_user_true['seq_id'] = 1
        test_user_true['ts_id'] = np.arange(seq_len)

        train_user_list.append(train_user)
        test_user_list.append(test_user_true.values)

        for i in range(1, test_user_size):
            seq_id = i + 1

            test_user_noise = test_user_true.values
            # sample item
            test_user_noise[seq_len - 1, 1] = np.random.randint(0, n_items)
            # update as noise sample
            test_user_noise[seq_len - 1, 2] = 0.0
            # update sequence id
            test_user_noise[:, 4] = seq_id
            test_user_list.append(test_user_noise)

            progress = 100*(prog_cntr / (n_users*test_user_size))
            print("progress: {:.2f}%".format(progress), end="\r")
            prog_cntr += 1


    train = pd.concat(train_user_list, axis=0)
    test = np.concatenate(test_user_list, axis=0)

    test = pd.DataFrame(test, columns=cols + ["ts_id"])

    X_test = test.pivot_table(index=['user_id', 'seq_id'], columns='ts_id', values='item_id').reset_index()
    y_test = test.pivot_table(index=['user_id', 'seq_id'], columns='ts_id', values='rating').reset_index()

    return train[['user_id', 'item_id', 'timestamp']], X_test, train['rating'].to_frame(), y_test