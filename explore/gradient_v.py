import os
import sys
from random import sample
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
from sklearn.manifold import TSNE

from preprocessing.meta_data_item_mapper import MetaDataMap, get_meta_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.utils import read_train_test_dir, get_mrs_arr, get_supp_k, get_comp_k
import torch
import pandas as pd
import numpy as np
from model.trainer import NeuralUtilityTrainer
from model._loss import loss_mse
from sklearn.preprocessing import OneHotEncoder
from preprocessing.utils import load_dict_output

Vector = numpy.ndarray
config = get_meta_config()


class Item:
    def __init__(self, asin: str, idx: int, title: str, vector: Vector, cat=None, bought=None) -> None:
        if cat is None:
            cat = []
        try:
            if "var aPage" in title:
                self.title = "omitted html"
            else:
                self.title = title
        except:
            self.title = "omitted html"
        self.cat = cat
        self.bought = bought
        self.asin = asin
        self.idx = idx
        self.vector = vector

    def get_cat(self):
        if self.cat and len(self.cat) > 1:
            return self.cat[1]
        else:
            return 'na'


class ItemHolder:
    def __init__(self, items: List[Item], id_map, catmap, boughtmap):
        self.items = items
        self.id_map = id_map
        self.cat_map = catmap
        self.bougt_map = boughtmap

    def item(self, asin: str) -> Item:
        return next(i for i in self.items if asin == i.asin)

    def item_for_idx(self, idx):
        return self.item(self.id_map[str(idx)])


class GradientHolder:
    def __init__(self, gradients, ih: ItemHolder):
        self.gradients = gradients
        self.items = ih
        self.mrs = get_mrs_arr(self.gradients)

    def get_supp_and_comp(self, asin: str, k=5):

        I = self.items.item(asin)
        item_idx = I.idx

        item = self.mrs[item_idx, :]
        print(item)
        supp = get_supp_k(item, k=k)
        comp = get_comp_k(item, k=k)

        print("Item: {}".format(I.title))
        print("Supplements: ")
        s = []
        c = []
        for i in supp:
            ii = self.items.item_for_idx(i)
            s.append(ii)
            print(ii.title, self.gradients[i])
        print("Complements: ")
        for i in comp:
            ii = self.items.item_for_idx(i)
            c.append(ii)
            print(self.items.item_for_idx(i).title, self.gradients[i])

        print(np.max(self.gradients))
        print(np.min(self.gradients))

        return (s, c)


def print_tuple(tup):
    return tup[0] + " " + str(tup[1])


def load_model_for(name, size=None, map_location=torch.device('cpu')):
    model_dir = config[name]['model']
    if size is None:
        size = ''
    else:
        size = '_' + size
    model = torch.load(model_dir + '/item_encoder_amazon_utility' + size + '_done.pt', map_location=map_location)
    return model


def find_item(items: List[Item], asin: str) -> Item:
    return next(i for i in items if asin == i.asin)


def print_is(iis):
    for i in iis:
        print(i[1].title + "  " + i[1].asin)


def load_items(name):
    df = pd.read_csv(config[name]['pp'] + "/x_train.csv")
    df = df['item_id']
    # dedup
    items = set(df.values.tolist())
    return items


def create_items(meta_name, size=None):
    mm = MetaDataMap(get_meta_config())
    nm = mm.get_all()

    model = load_model_for(meta_name, size)
    weights = model.embedding.weights.weight.data.to('cpu')
    item_map = load_items(meta_name)
    catmap = mm.get_cat(meta_name)
    boughtmap = mm.get_bought(meta_name)
    items = []
    for val in item_map:
        vec = weights[:, int(val)]
        asin = nm[str(int(val))]
        title = nm[nm[str(int(val))]]
        if asin in catmap:
            cat = catmap[asin]
        else:
            cat = []
        if asin in boughtmap:
            b = boughtmap[asin]
        else:
            b = []
        items.append(Item(asin=asin, idx=int(val), vector=vec, title=title, cat=cat, bought=b))
    return ItemHolder(items, mm.get_id_item_map(meta_name), catmap, boughtmap)


def dim_reduce(items: List[Item]):
    print('H DIM Size ' + str(items[0].vector.shape[0]))
    arr = numpy.empty((0, items[0].vector.shape[0]))

    labels = []
    i = 0
    items = sample(items, config['num_items'])

    for item in items:
        vec = item.vector
        arr = numpy.append(arr, [vec.numpy()], axis=0)
        labels.append(item.get_cat())

    unq_labels = set(labels)
    N = len(unq_labels)
    label_color_map = {}
    cc = []
    for idx, label in enumerate(unq_labels):
        label_color_map[label] = idx
        cc.append(label)

    mapped = []
    for l in labels:
        mapped.append(label_color_map[l])

    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom CMAP', cmaplist, cmap.N)
    bounds = numpy.linspace(0, N, N + 1)

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    scat = plt.scatter(x_coords, y_coords, c=mapped, cmap=cmap, norm=norm)

    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)

    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_ticklabels(cc)
    cb.set_label('Custom cbar')
    ax.set_title('Categories')

    plt.show()
    return Y


def do_gradients(meta_name, size=None):
    model = load_model_for(meta_name)

    # Process data
    X_train, X_test, y_train, y_test = read_train_test_dir(config[meta_name]['pp'])
    stats = load_dict_output(config[meta_name]['pp'], "stats.json")

    one_hot = OneHotEncoder(categories=[range(stats["n_items"])])
    items_for_grad = one_hot.fit_transform(np.arange(stats["n_items"]).reshape(-1, 1)).todense().astype(np.float32)

    train = np.concatenate([X_train, y_train], axis=1)
    test = np.concatenate([X_test, y_test], axis=1)

    X = np.concatenate((train, test), axis=0)

    df = pd.DataFrame(X, columns=["user_id", "item_id", "rating"])
    item_means = df[['item_id', 'rating']].groupby("item_id").mean()

    users = None
    items_for_grad = torch.from_numpy(items_for_grad)
    y_true = torch.from_numpy(item_means.values.reshape(-1, 1))

    gradients = NeuralUtilityTrainer.get_gradient(model, loss_mse, users, items_for_grad, y_true)
    print(gradients)

    return gradients


def create_gr(meta, size=None):
    ih = create_items(meta, size)
    g = do_gradients(meta, size)
    return GradientHolder(g, ih)


g_512 = create_gr('grocery', str(512))

# g_1000 = create_items('grocery')
# hk_256 = create_items('home_kitchen')
