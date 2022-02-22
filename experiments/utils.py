import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime
import math
from scipy import stats

def get_eval_metrics(output, at_k=5):

    output.sort_values(by=['user_id', 'pred'], inplace=True, ascending=False)
    output = output.groupby('user_id').head(at_k)
    output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first', ascending=False).astype(float)
    output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)

    rmse = np.sqrt(mean_squared_error(output.y_true, output.pred))



    avg_dcg = output.dcg.mean()

    return output, rmse, avg_dcg


def get_eval_metrics_sequential(users_test, preds, y_test, seq_len, eval_k):

    pred_cols = ["pred_{}".format(x) for x in range(seq_len)]
    true_cols = ["y_true_{}".format(x) for x in range(seq_len)]

    output = pd.DataFrame(np.concatenate((users_test, preds, y_test), axis=1),
                          columns=['user_id'] + pred_cols + true_cols)

    pred_long = pd.melt(output[['user_id'] + pred_cols], id_vars='user_id', value_vars=pred_cols, value_name='pred')
    true_long = pd.melt(output[['user_id'] + true_cols], id_vars='user_id', value_vars=true_cols, value_name='y_true')

    output = pd.concat([pred_long[['user_id', 'pred']], true_long['y_true']], axis=1)

    output, rmse, dcg = get_eval_metrics(output, at_k=eval_k)

    return output, rmse, dcg


def get_idcg(k):
    ideal = np.zeros(k)
    ideal[0] = 1
    rank = np.arange(1, k+1)

    idcg = (np.power(2, ideal) - 1) / np.log2(rank + 1)
    return idcg


def get_choice_eval_metrics(output, at_k=5):

    output.sort_values(by=['user_id', 'pred'], inplace=True, ascending=False)
    output = output.groupby('user_id').head(at_k)


    output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first', ascending=False).astype(float)
    output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)

    results = output[['user_id', 'y_true', 'dcg']].groupby("user_id").sum().mean()



    ndcg = results['dcg']
    hit_ratio = results['y_true']


    return output, hit_ratio, ndcg


def get_choice_eval_sequential(output, at_k=5):

    output.sort_values(by=['user_id', 'pred'], inplace=True, ascending=False)
    output = output.groupby('user_id').head(at_k)


    output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first', ascending=False).astype(float)
    output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)

    results = output[['user_id', 'y_true', 'dcg']].groupby("user_id").sum().mean()



    ndcg = results['dcg']
    hit_ratio = results['y_true']


    return output, hit_ratio, ndcg


def get_choice_eval_metrics_sequential(users_test, preds, y_test, seq_len, eval_k):

    #pred_cols = ["pred_{}".format(x) for x in range(seq_len)]
    #true_cols = ["y_true_{}".format(x) for x in range(seq_len)]

    y_test_ts = y_test[:, seq_len-1].reshape(-1,1)
    preds_ts = preds[:, seq_len-1].reshape(-1,1)


    output = pd.DataFrame(np.concatenate((users_test, preds_ts, y_test_ts), axis=1),
                          columns=['user_id', 'pred', 'y_true'])

    output, hit_ratio, ndcg = get_choice_eval_sequential(output, at_k=eval_k)

    return output, hit_ratio, ndcg


def get_test_batch_size(n):

    b = 50

    while n % b > 0:
        b -= 1

    return b

def get_test_sample_size(n, k):

    floor = n // k
    n_update = floor*k

    return n_update


def read_train_test_dir(dir, drop_ts=True):

    x_train = pd.read_csv(dir + "/x_train.csv")
    x_test = pd.read_csv(dir + "/x_test.csv")

    if drop_ts:
        x_train = x_train[['user_id', 'item_id']].values.astype(np.int64)
        x_test = x_test[['user_id', 'item_id']].values.astype(np.int64)
    else:
        x_train = x_train.values.astype(np.int64)
        x_test = x_test.values.astype(np.int64)

    y_train = pd.read_csv(dir + "/y_train.csv").values.reshape(-1,1).astype(np.float32)
    y_test = pd.read_csv(dir + "/y_test.csv").values.reshape(-1,1).astype(np.float32)



    return x_train, x_test, y_train, y_test


def log_output(out_dir, model_name, params, output):

    log_dir = out_dir + "/log"

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now = datetime.now()
    fname = "{}/{}-{}.txt".format(log_dir, model_name, now)

    with open(fname, 'w') as f:
        f.write("{} - {}\n".format(model_name, now))
        for name, val in params.items():
            f.write("{}: {}\n".format(name, val))

        for i in output:
            f.write("{:.4f}\n".format(i))



def read_train_test_dir_sequential(dir):

    x_train = pd.read_csv(dir + "/x_train.csv")
    x_train = x_train.values.astype(np.int64)


    x_test = pd.read_csv(dir + "/x_test.csv")


    y_train = pd.read_csv(dir + "/y_train.csv").values.reshape(-1,1).astype(np.float32)
    y_test = pd.read_csv(dir + "/y_test.csv")



    return x_train, x_test, y_train, y_test


def compute_pariwise_mrs(grad):

    n = len(grad)
    mrs_mat = np.zeros((n, n))

    for i, g_i in enumerate(grad):
        for j, g_j in enumerate(grad):

            if g_j == 0.0:
                g_j = 1e-3

            mrs_mat[i, j] = - (g_i / g_j)

    return mrs_mat

def get_mrs_arr(grad):

    return np.sum.outer(grad, 1/grad)

def mrs_error(M1, M2):

    assert M1.shape == M2.shape

    n = M1.shape[0]

    idx = np.triu_indices(n)

    m1 = M1[idx]
    m2 = M2[idx]

    mse = math.sqrt(mean_squared_error(m1, m2))
    return mse

def get_analytical_cobb_douglas_mrs(w1, w2):

    #return w1 + w2
    return -w1/w2

def get_analytical_stone_geary_mrs(w1, w2, gamma_1, gamma_2):

    return -(w1*gamma_1)/(w2*gamma_2)

def get_analytical_ces_mrs(w1, w2, rho):

    return - (w1 / w2)**(rho-1)

def get_mrs_mat(x, w, mrs_func, rho=None):
    n = x.shape[0]
    mrs_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            if rho is None:
                mrs_mat[i, j] = mrs_func(w1=w[i], w2=w[j])
            else:
                mrs_mat[i, j] = mrs_func(w1=w[i], w2=w[j], rho=rho)

    return mrs_mat




def logit(x):
    return 1 / (1 + np.exp(-x))


def cobb_douglas(x, w):
    # TODO: Think more about this. This assumes that we are already at one. What's the utility at 2?
    eps = 1.0
    log_x = np.log(x + eps)
    log_u = np.dot(log_x, w) + np.random.normal(0, 1, 1)[0]
    u = np.exp(log_u)

    return u

def ces(x, w, rho):

    x_power = np.power(x, rho)
    inner_prod = np.dot(x_power, w)
    u = np.power(inner_prod, rho)
    return u


def get_supp_k(arr, k):
    return np.argsort(arr)[-k:]

def get_comp_k(arr, k):
    return np.argsort(arr)[:k]

def permutationTest(x,y,nperm, method = 'mean', twoSided=True):

    np.random.seed(seed=None)
    x_array = np.array(x)
    y_array = np.array(y)
    n = len(x_array)
    perm_ts = np.zeros(nperm)
    all_obs = np.concatenate([x_array,y_array])

    # Permutation Test: Difference of Means #
    if method == 'mean':
        # If twoSided is True; Perform two sided hypothesis test #
        if twoSided:
            obs_diff = abs(np.mean(x_array) - np.mean(y_array))
        # Otherwise perform one sided test: Ho: median X = median Y, Ha: median X > median Y
        else:
            obs_diff = np.mean(x_array) - np.mean(y_array)

        for i in range(nperm):
            np.random.shuffle(all_obs)
            perm_x = all_obs[:n]
            perm_y = all_obs[n:]
            if twoSided:
                perm_ts[i] = abs(np.mean(perm_x) - np.mean(perm_y))
            else:
                perm_ts[i] = np.mean(perm_x) - np.mean(perm_y)

    # Permutation Test: Difference of Medians #

    elif method == 'median':
        # If twoSided is True; Perform two sided hypothesis test #
        if twoSided:
            obs_diff = abs(np.median(x_array) - np.median(y_array))
        # Otherwise perform one sided test: Ho: median X = median Y, Ha: median X > median Y
        else:
            obs_diff = np.median(x_array) - np.median(y_array)

        for i in range(nperm):
            np.random.shuffle(all_obs)
            perm_x = all_obs[:n]
            perm_y = all_obs[n:]
            if twoSided:
                perm_ts[i] = abs(np.median(perm_x) - np.median(perm_y))
            else:
                perm_ts[i] = np.median(perm_x) - np.median(perm_y)
    else:
        raise Exception('Method for permutation must be mean or median')

    pval = len(perm_ts[perm_ts > obs_diff])/nperm
    return pval


def diff_means_t_stat(x_1, x_2, s_1, s_2, n):
    """
    Difference of means t stat (https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test)
        - the two sample sizes (that is, the number n of participants of each group) are equal;
        - it can be assumed that the two distributions have the same variance;
    :param x_1: sample mean group 1
    :param x_2: sample mean group 2
    :param s_1: std of group 1
    :param s_2: std of group 2
    :param n: sample size
    :return:
    """

    std_pool = math.sqrt((s_1**2 + s_2**2) / 2)
    t = (x_1 - x_2) / (std_pool * math.sqrt(2/n))

    #t =  (x_1 - x_2) / math.sqrt(s_1**2 / n + s_2**2 / n)

    df = 2 * n - 2
    p = (1 - stats.t.cdf(t, df=df))*2

    return np.round(t,4), np.round(p,4)