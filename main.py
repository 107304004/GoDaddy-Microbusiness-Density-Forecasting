import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
# import models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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
1 (add feature: microbusiness_density 1 lag)
--cfips
0 (nothing)
1 (add feature: microbusiness_density_mean (groupby cfips) )
2 (add:feature: microbusiness_density_std (groupby cfips) )
3 (add feature: trend (linear regression slope) )
4 (add all features above)
--active
0 (nothing)
1 (add features: active & adult_pop)
'''

def cut_valid(df, date):

    train = df.loc[df['first_day_of_month']<date]
    val = df.loc[df['first_day_of_month']>date]
    print('train shape: ', train.shape)
    print('val shape: ', val.shape)

    return train, val


def fe_shift(df, lag=1):

    df[f'mbd_lag_{lag}'] = df.groupby('cfips')['microbusiness_density'].shift(lag)
    features.append(f'mbd_lag_{lag}')

    return df.loc[df[f'mbd_lag_{lag}'].notnull()]


def fe_cfips(train, test, mean=0, std=0, trend=0):

    if mean > 0:
        train = train.merge(train.groupby('cfips').mean()['microbusiness_density'],
                            on='cfips', how='left', suffixes=('', '_mean'))
        test = test.merge(train.groupby('cfips').mean()['microbusiness_density'],
                            on='cfips', how='left', suffixes=('', '_mean'))
        features.append('microbusiness_density_mean')

    if std > 0:
        train = train.merge(train.groupby('cfips').std()['microbusiness_density'],
                            on='cfips', how='left', suffixes=('', '_std'))
        test = test.merge(train.groupby('cfips').std()['microbusiness_density'],
                            on='cfips', how='left', suffixes=('', '_std'))
        features.append('microbusiness_density_std')

    if trend > 0:
        cfips = test['cfips'].unique()
        n_train = len(train.loc[train['cfips']==cfips[0]])
        x_train = np.arange(n_train).reshape((-1,1))

        for c in cfips:
            y_train = train.loc[train['cfips']==c]
            ## predict micro
            model = LinearRegression()
            model.fit(x_train, y_train['microbusiness_density'])
            # get_coef
            coef = model.coef_[0]
            # if int(c) < 1010:
            #     print(coef)
            train.loc[train['cfips']==c, 'trend'] = coef
            test.loc[test['cfips']==c, 'trend'] = coef
        features.append('trend')

    return train, test


def fe_active(train, test):

    cfips = test['cfips'].unique()
    n_train = len(train.loc[train['cfips']==cfips[0]])
    n_test = len(test.loc[test['cfips']==cfips[0]])

    x_train = np.arange(n_train).reshape((-1,1))
    x_test = np.arange(n_train-1, n_train+n_test).reshape((-1,1))

    for c in cfips:

        y_train = train.loc[train['cfips']==c]

        ## predict active
        model = LinearRegression()
        model.fit(x_train, y_train['active'])
        pred = model.predict(x_test)

        # shift
        last_active = y_train['active'].iloc[-1]
        shift = pred[0] - last_active
        pred = pred - shift

        # assign new active for test
        test.loc[test['cfips']==c, 'active'] = pred[1:]

        ## predict adult_pop
        adult_pop = []
        last_year = y_train['year'].iloc[-1]
        last_adult_pop = y_train['adult_pop'].iloc[-1]
        ratio = y_train.groupby('year').mean()['adult_pop'].iloc[-1] /  y_train.groupby('year').mean()['adult_pop'].iloc[-2]
        # if int(c) < 1010:
        #     print(ratio)
        for y in test.loc[test['cfips']==c, 'year']:
            if last_year == y:
                adult_pop.append(last_adult_pop)
            else:
                adult_pop.append(last_adult_pop * ratio)

        # assign new adult_pop for test
        test.loc[test['cfips']==c, 'adult_pop'] = adult_pop

    features.append('active')
    features.append('adult_pop')

    return train, test

# preparing data
data = pd.read_csv('train_0224_v1.csv')
# data = data.loc[data['first_day_of_month']>'2020-02']

features = ['year', 'month', 'pct_bb', 'pct_college',
        'pct_foreign_born', 'pct_it_workers', 'median_hh_inc']

target = ['microbusiness_density']

if args.shift > 0:
    print('Preparing shift feature...')
    data = fe_shift(data, 1)
    data = fe_shift(data, 2)
    data = fe_shift(data, 3)
    data = fe_shift(data, 4)
    data = fe_shift(data, 5)
    data = fe_shift(data, 6)

train, val = cut_valid(data, args.cutval_date)

if args.cfips > 0:
    print('Preparing cfips feature...')
    train, val = fe_cfips(train, val, 1, 1, 1)

if args.active > 0:
    print('Preparing new active & adult_pop...')
    train, val = fe_active(train, val)


# training
print('')
print('Training Features: ', features)
# scaler = StandardScaler()
# scaler = MinMaxScaler()

def train_xgb():
    print('')
    print('Model: ', 'xgbregressor')
    xgb = XGBRegressor()
    # xgb.fit(scaler.fit_transform(train[features]), train[target])
    xgb.fit(train[features], train[target])
    # prediction = xgb.predict(scaler.fit_transform(val[features]))
    prediction = xgb.predict(val[features])
    return prediction

def train_lgbm():
    print('')
    print('Model: ', 'lgbmregressor')
    lgbm = LGBMRegressor()
    lgbm.fit(train[features], train[target])
    prediction = lgbm.predict(val[features])
    return prediction


# eval
def SMAPE(a, f):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)

score = SMAPE(val[target].values.reshape(-1), train_xgb())
# print('')
print('smape score: ', round(score, 4))

score = SMAPE(val[target].values.reshape(-1), train_lgbm())
# print('')
print('smape score: ', round(score, 4))


