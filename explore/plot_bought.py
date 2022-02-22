import os
import sys
from random import sample
from typing import List

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import torch
from numpy.linalg import norm
from scipy import spatial
from sklearn.manifold import TSNE

from preprocessing.meta_data_item_mapper import MetaDataMap, get_meta_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


class Answer:
    def __init__(self, item: Item, score: float) -> None:
        self.item = item
        self.score = score


def dot_product(a: Vector, b: Vector) -> float:
    return norm(a) * norm(b)


def cosine_similarity(a: Vector, b: Vector) -> float:
    return 1 - spatial.distance.cosine(a, b)


class ItemHolder:
    def __init__(self, items: List[Item], id_map, catmap, boughtmap):
        self.items = items
        self.id_map = id_map
        self.cat_map = catmap
        self.bought_map = boughtmap

    def item(self, asin: str) -> Item:
        return next((i for i in self.items if asin == i.asin), None)

    def item_for_idx(self, idx):
        return self.item(self.id_map[str(idx)])

    def get_also_bought(self, asin: str):
        item = self.item(asin)
        bought = []
        for b in item.bought:
            bought.append(self.item(b))
        return (item, bought)


def load_model_for(name, size=None, map_location=torch.device('cpu')):
    model_dir = config[name]['model']
    if size is None:
        size = ''
    else:
        size = '_' + size
    model = torch.load(model_dir + '/item_encoder_amazon_utility' + size + '_done.pt', map_location=map_location)
    return model


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


def dim_reduce(og: Item, answers: List[Answer], others: List[Answer]):
    print('H DIM Size ' + str(answers[0].item.vector.shape[0]))
    arr = numpy.empty((0, answers[0].item.vector.shape[0]))

    labels = []

    vec = og.vector
    arr = numpy.append(arr, [vec.numpy()], axis=0)
    labels.append(og.asin)
    scores = [1.0]
    for answer in answers:
        vec = answer.item.vector
        arr = numpy.append(arr, [vec.numpy()], axis=0)
        labels.append(answer.item.asin)
        scores.append(2.0)

    for answer in others:
        vec = answer.item.vector
        arr = numpy.append(arr, [vec.numpy()], axis=0)
        labels.append(answer.item.asin)
        scores.append(3.0)

    tsne = TSNE(n_components=2, random_state=0, perplexity=40, init="pca", n_iter=2500)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    plt.scatter(x_coords, y_coords, c=scores, cmap='viridis')

    plt.xlim(x_coords.min() + 10, x_coords.max() + 10)
    plt.ylim(y_coords.min() + 10, y_coords.max() + 10)

    ax.set_title('Categories')

    plt.show()
    return Y


class BoughtPlotter:
    def __init__(self, ih: ItemHolder):
        self.items = ih

    def get_also_bought(self, asin: str, k=5):
        item, bought = self.items.get_also_bought(asin)
        answers = []
        for b in bought:
            if b is not None:
                score = cosine_similarity(item.vector, b.vector)
                answers.append(Answer(b, score))
        answers = sorted(answers, key=lambda t: t.score, reverse=False)
        return (item, answers)

    def score_others(self, item: Item, others: List[Item]):
        answers = []
        for o in others:
            score = cosine_similarity(item.vector, o.vector)
            answers.append(Answer(o, score))
        return answers

    def plot_also_bought(self, asin: str):

        item, answers = self.get_also_bought(asin)

        other_items = sample(self.items.items, config['num_items'])
        other_answers = self.score_others(item, other_items)

        dim_reduce(item, answers, other_answers)
        return item, answers, other_answers


def create_plotter(meta, size=None):
    ih = create_items(meta, size)
    return BoughtPlotter(ih)

# plotter_256 = create_plotter('home_kitchen', str(256))
# plotter_768 = create_plotter('home_kitchen', str(768))
# plotter_1000 = create_plotter('home_kitchen')?
# g_1000 = create_items('grocery')
# hk_256 = create_items('home_kitchen')
# i_512 = plotter_512.get_also_bought('B00ODEN5M4')
# i_1000 = plotter_1000.get_also_bought('B00ODEN5M4')
# i_256 = plotter_256.get_also_bought('B00ODEN5M4')
#
# b_512 = i_512[1]
# b_256 = i_256[1]
# a_1024 = i_1000[1]

# jelly = item('B00ODEN5M4')
# pb = item('B00061ENVK')
# bq = item('B0028B9ZGE')
# sy = item('B0005ZZADW')
#
# spg = item('B000G0K112')
