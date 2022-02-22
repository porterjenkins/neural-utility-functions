import os
import sys

import numpy
import pandas as pd
import torch

from preprocessing.meta_data_item_mapper import MetaDataMap, get_meta_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

Vector = numpy.ndarray
config = get_meta_config()


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


def create_categories(meta_name):
    mm = MetaDataMap(get_meta_config())
    nm = mm.get_all()

    item_map = load_items(meta_name)
    catmap = mm.get_cat(meta_name)

    categories = {}
    categories['na'] = []
    for val in item_map:
        asin = nm[str(int(val))]
        if asin in catmap:
            cat = catmap[asin]
        else:
            cat = []
        for category in cat:
            if category in categories:
                categories[category].append(asin)
            else:
                categories[category] = [asin]

    return categories


def create_cat_vectors(meta_name, size=None):
    mm = MetaDataMap(get_meta_config())

    model = load_model_for(meta_name, size)
    weights = model.embedding.weights.weight.data.to('cpu')

    cats = create_categories(meta_name)
    asin_to_idx = mm.get_asin_idx_map(meta_name)
    cat_vec = {}
    for category in cats.keys():
        asins = cats[category]
        arr = numpy.empty((0, int(size)))
        for asin in asins:
            if asin in asin_to_idx:
                idx = asin_to_idx[asin]
                vec = weights[:, int(idx)]
                arr = numpy.append(arr, [vec.numpy()], axis=0)
        mean = numpy.average(arr, axis=0)
        cat_vec[category] = mean
    return cats, cat_vec


c, cv = create_cat_vectors('grocery', str(512))
