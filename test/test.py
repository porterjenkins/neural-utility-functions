import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import gzip
import pandas as pd

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


df = pandas_df(cfg.vals['amazon_dir'] + "/Grocery_and_Gourmet_Food_5.json.gz")
df = df.drop(columns=[
    "reviewerName",
    "reviewText",
    "summary",
    #"unixReviewTime",
    "reviewTime",
    "verified",
    "vote",
    "style",
    "image"
])
sample_fraction = .1
df = df.sample(frac=sample_fraction).reset_index(drop=True)


print(df)
print(df.columns)