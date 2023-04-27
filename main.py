import numpy as np
import pandas as pd

from feature_engineering import fe_active, fe_cfips, fe_shift
from sklearn.preprocessing import MinMaxScaler
import torch
# import models
from models import train_dl, train_xgb, train_lgbm
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--cutval_date', '-date', type=str) # cut valid date
parser.add_argument('--shift', '-shift', type=int)
parser.add_argument('--cfips', '-cfips', type=int)
parser.add_argument('--active', '-active', type=int)
# args.cutval_date / args.shift / args.cfips / args.active
args = parser.parse_args()

'''
--cutval_date
2022-12 (val: 2022-12~)
--shift
0 (nothing)
1 (add feature: microbusiness_density 1~6 lag)
--cfips
0 (nothing)
1 (add features: mbd_mean, mbd_std, mbd_trend)
--active
0 (nothing)
1 (add features: active & adult_pop)
'''


# preparing data
data = pd.read_csv('train_0224_v1.csv')
# data = data.loc[data['first_day_of_month']>'2020-02']

def cut_valid(df, date):

    train = df.loc[df['first_day_of_month']<date]
    val = df.loc[df['first_day_of_month']>date]
    print('train shape: ', train.shape)
    print('val shape: ', val.shape)

    return train, val


features = ['year', 'month', 'pct_bb', 'pct_college',
        'pct_foreign_born', 'pct_it_workers', 'median_hh_inc']

target = ['microbusiness_density']

# feature engineering
if args.shift > 0:
    print('Preparing shift feature...')
    print('Data shape: ', data.shape)
    data, features = fe_shift(data, features, 24)

train, val = cut_valid(data, args.cutval_date)

if args.cfips > 0:
    print('Preparing cfips feature...')
    train, val, features = fe_cfips(train, val, features, 1, 1, 1)

if args.active > 0:
    print('Preparing new active & adult_pop...')
    train, val, features = fe_active(train, val, features)

# features = features[7:]


# deep learning training settings
batch_size = 64

# scaler df to tensor
scaler = MinMaxScaler()
train_for_dl_fe = scaler.fit_transform(train[features])
val_for_dl_fe = scaler.fit_transform(val[features])
x, y = torch.Tensor(train_for_dl_fe), torch.Tensor(train[target].values)
val_x, val_y = torch.Tensor(val_for_dl_fe), torch.Tensor(val[target].values)

# Dataset
train_dataset = TensorDataset(x, y)
val_dataset = TensorDataset(val_x, val_y)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

n_features = len(features)

# eval
def SMAPE(a, f):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)

# training
print('')
print('Training Features: ', features)

score = SMAPE(val[target].values.reshape(-1), train_xgb(train, val, features, target))
print('smape score: ', round(score, 4))

score = SMAPE(val[target].values.reshape(-1), train_lgbm(train, val, features, target))
print('smape score: ', round(score, 4))

p, v = train_dl(train_loader, val_loader, features, 'nn', n_epochs=10)
p = torch.cat([t.view(-1) for t in p]).numpy()
v = torch.cat([t.view(-1) for t in v]).numpy()
print('smape score: ', round(SMAPE(p, v), 5))

p, v = train_dl(train_loader, val_loader, features, 'lstm')
p = torch.cat([t.view(-1) for t in p]).numpy()
v = torch.cat([t.view(-1) for t in v]).numpy()
print('smape score: ', round(SMAPE(p, v), 5))

p, v = train_dl(train_loader, val_loader, features, 'gru')
p = torch.cat([t.view(-1) for t in p]).numpy()
v = torch.cat([t.view(-1) for t in v]).numpy()
print('smape score: ', round(SMAPE(p, v), 5))


