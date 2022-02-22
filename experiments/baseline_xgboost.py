import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from generator.generator import Generator
from preprocessing.utils import split_train_test_user, get_one_hot_encodings
from model.utils import load_embedding, embedding_to_df

RANDOM_SEED = 1990


embedding = embedding_to_df(load_embedding(fname=cfg.vals['model_dir']+'/embedding.txt'))

df = pd.read_csv(cfg.vals['movielens_dir'] + "/preprocessed/ratings.csv")


#X = df[['user_id']]
y = df['rating']

X_sparse = get_one_hot_encodings(df['item_id'])
X_sparse['user_id'] = df['user_id']

X_dense = pd.merge(df[['user_id', 'item_id']], embedding, left_on='item_id', right_on='id', how='left')

X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = split_train_test_user(X_sparse, y, random_seed=RANDOM_SEED)
X_train_dense, X_test_dense, y_train_dense, y_test_dense = split_train_test_user(X_dense, y, random_seed=RANDOM_SEED)


X_train_sparse = X_train_sparse.drop(['user_id', 'item_id'], axis=1).values
X_train_dense = X_train_dense.drop(['user_id', 'item_id', 'id'], axis=1).values

x_test_user = X_test_dense['user_id'].copy().values

X_test_sparse = X_test_sparse.drop(['user_id', 'item_id'], axis=1).values
X_test_dense = X_test_dense.drop(['user_id', 'item_id', 'id'], axis=1).values


# train model with sparse item vectors
xg_reg_sparse = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10)
print("training...")
xg_reg_sparse.fit(X_train_sparse, y_train_sparse)
print("complete")

preds = xg_reg_sparse.predict(X_test_sparse)

output = pd.DataFrame(np.concatenate([x_test_user.reshape(-1,1), preds.reshape(-1,1), y_test_sparse.values.reshape(-1,1)], \
                                    axis=1), columns = ['user_id', 'pred', 'y_true'])

output.sort_values(by=['user_id', 'pred'], inplace=True, ascending=False)
output = output.groupby('user_id').head(5)
output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first', ascending=False).astype(float)
output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)

rmse = np.sqrt(mean_squared_error(output.y_true, output.pred))

print(output)

avg_dcg = output.dcg.mean()
print(rmse)
print(avg_dcg)


# Train dense with dense item vectors
xg_reg_dense = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10)


print("training...")
xg_reg_dense.fit(X_train_dense, y_train_dense)
print("complete")

preds = xg_reg_dense.predict(X_test_dense)

output = pd.DataFrame(np.concatenate([x_test_user.reshape(-1,1), preds.reshape(-1,1), y_test_dense.values.reshape(-1,1)], \
                                    axis=1), columns = ['user_id', 'pred', 'y_true'])

output.sort_values(by=['user_id', 'pred'], inplace=True, ascending=False)
output = output.groupby('user_id').head(5)
output['rank'] = output[['user_id', 'pred']].groupby('user_id').rank(method='first', ascending=False).astype(float)
output['dcg'] = (np.power(2, output['y_true']) - 1) / np.log2(output['rank'] + 1)



rmse = np.sqrt(mean_squared_error(output.y_true, output.pred))

print(output)

avg_dcg = output.dcg.mean()
print(rmse)
print(avg_dcg)