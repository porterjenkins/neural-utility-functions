import os
import pickle
import sys

import config.config as cfg
from preprocessing.utils import load_dict_output, pandas_df

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

wanted_columns = ['category', 'title', 'also_buy', 'asin']

META_DATA_CONFIG = {
    'metas': ['pantry', 'home_kitchen', 'grocery'],
    'num_items': 1000,
    'dir': cfg.vals['amazon_dir'],
    'meta_dir': cfg.vals['amazon_dir'] + "meta/",
    'pantry': {
        'pp': cfg.vals['amazon_dir'] + "/preprocessed_pantry/",
        'name': 'Prime_Pantry',
        'model': cfg.vals['model_dir'] + "/pantry_ahmad/"
    },
    'home_kitchen': {
        'pp': cfg.vals['amazon_dir'] + "/preprocessed_home_kitchen/",
        'name': 'Home_and_Kitchen',
        'model': cfg.vals['model_dir'] + "/home_kitchen_ahmad/"

    },
    'grocery': {
        'pp': cfg.vals['amazon_dir'] + "/preprocessed_grocery/",
        'name': "Grocery_and_Gourmet_Food",
        'model': cfg.vals['model_dir'] + "/grocery_ahmad/"
    }
}


def create_asin_category_mapping(name, pp_data_dir, meta_df):
    records = meta_df.to_dict('record')
    asin_mapping = dict()
    name = name
    for record in records:
        asin_mapping[record['asin']] = record['category']
    with open(pp_data_dir + '/' + name + '_meta_id_cat_map.pt', 'wb') as f:
        pickle.dump(asin_mapping, f)


def create_title_id_meta_mapping(name, pp_data_dir, meta_df):
    records = meta_df.to_dict('record')
    item_mapping = load_dict_output(pp_data_dir, 'id_item_map.json')
    asin_mapping = dict()
    name = name
    for record in records:
        asin_mapping[record['asin']] = record['title']

    item_mapping.update(asin_mapping)
    with open(pp_data_dir + '/' + name + '_meta_id_item_mapping.pt', 'wb') as f:
        pickle.dump(item_mapping, f)


def create_asin_bought_mapping(name, pp_data_dir, meta_df):
    records = meta_df.to_dict('record')
    asin_mapping = dict()
    name = name
    for record in records:
        asin_mapping[record['asin']] = record['also_buy']
    with open(pp_data_dir + '/' + name + '_meta_id_bought_mapping.pt', 'wb') as f:
        pickle.dump(asin_mapping, f)


def do_all_meta_mapping(name, config):
    meta_data_file = config['meta_dir'] + 'meta_' + config[name]['name'] + ".json.gz"
    pp_data_dir = config[name]['pp']
    meta_df_raw = load_meta_df(meta_data_file)
    columns = meta_df_raw.columns.to_list()

    mapping_cols = ['title', 'asin']
    title_id_df = meta_df_raw[mapping_cols]
    create_title_id_meta_mapping(name, pp_data_dir, title_id_df)

    if 'category' in columns:
        print("Categories Found")
        mapping_cols = ['asin', 'category']
        cat_df = meta_df_raw[mapping_cols]
        create_asin_category_mapping(name, pp_data_dir, cat_df)
    else:
        print("No meta found")

    if 'also_buy' in columns:
        print("Also Bought Found")
        mapping_cols = ['asin', 'also_buy']
        b_df = meta_df_raw[mapping_cols].dropna()
        create_asin_bought_mapping(name, pp_data_dir, b_df)


def load_meta_df(m_file):
    print("getting data from {}".format(m_file))
    return pandas_df(m_file)


def get_meta_config():
    return META_DATA_CONFIG


def do_all_meta():
    config = get_meta_config()
    for cat in config['metas']:
        do_all_meta_mapping(cat, config)


class MetaDataMap:
    def __init__(self, config):
        self.config = config
        self.meta_files = dict()
        self.all_asin_to_titles = dict()

        for meta in self.config['metas']:
            data = self.config[meta]
            pp = data['pp']
            self.meta_files[meta] = {}
            self.meta_files[meta]['bought'] = pp + '/' + meta + '_meta_id_bought_mapping.pt'
            self.meta_files[meta]['id'] = pp + '/' + meta + '_meta_id_item_mapping.pt'
            self.meta_files[meta]['cat'] = pp + '/' + meta + '_meta_id_cat_map.pt'
            self.meta_files[meta]['idx'] = load_dict_output(pp, 'item_id_map.json')
            self.meta_files[meta]['id_item'] = load_dict_output(pp, 'id_item_map.json')

    def get_all(self):
        if len(self.all_asin_to_titles) == 0:
            all = dict()
            for meta in self.config['metas']:
                all.update(self.get_id_asin(meta))
            self.all_asin_to_titles = all

        return self.all_asin_to_titles

    def get_avail(self):
        return self.config['metas']

    def get_id_item_map(self, meta):
        return self.meta_files[meta]['id_item']

    def get_asin_idx_map(self, meta):
        return self.meta_files[meta]['idx']

    def get_id_asin(self, meta):
        meta_file = self.meta_files[meta]['id']
        return self.load_mapping_for(meta_file)

    def get_bought(self, meta):
        meta_file = self.meta_files[meta]['bought']
        return self.load_mapping_for(meta_file)

    def get_cat(self, meta):
        meta_file = self.meta_files[meta]['cat']
        return self.load_mapping_for(meta_file)

    def get_idx_for_asin(self, meta, asin):
        if asin in self.meta_files[meta]['idx']:
            return self.meta_files[meta]['idx'][asin]
        else:
            return None

    def load_mapping_for(self, file):
        from os import path
        if path.exists(file):
            with open(file, 'rb') as f:
                mm = pickle.load(f)
            return mm
        else:
            return {}
