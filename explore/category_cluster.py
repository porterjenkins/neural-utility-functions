from random import sample
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import torch
from sklearn.manifold import TSNE

from preprocessing.meta_data_item_mapper import MetaDataMap, get_meta_config

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


def print_tuple(tup):
    return tup[0] + " " + str(tup[1])


def load_model_for(name, size=None):
    model_dir = config[name]['model']
    if size is None:
        size = ''
    else:
        size = '_' + size
    model = torch.load(model_dir + '/item_encoder_amazon_utility' + size + '_done.pt')
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
    return items


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


g_512 = create_items('grocery', str(512))
g_1000 = create_items('grocery'
                      )
hk_256 = create_items('home_kitchen')
