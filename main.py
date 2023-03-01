import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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


def cut_valid(df, date):

    train = df.loc[df['first_day_of_month']<date]
    val = df.loc[df['first_day_of_month']>date]
    print('train shape: ', train.shape)
    print('val shape: ', val.shape)

    return train, val


def fe_shift(df, lag=1):

    print('Preparing shift feature...')
    df[f'mbd_lag_{lag}'] = df.groupby('cfips')['microbusiness_density'].shift(lag)
    features.append(f'mbd_lag_{lag}')

    return df.loc[df[f'mbd_lag_{lag}'].notnull()]


def fe_cfips(train, test, mean=0, std=0, trend=0):

    print('Preparing cfips feature...')
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

        for c in tqdm(cfips):
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

    print('Preparing new active & adult_pop...')
    for c in tqdm(cfips):

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
    data = fe_shift(data)

train, val = cut_valid(data, args.cutval_date)

if args.cfips == 1:
    train, val = fe_cfips(train, val, 1, 0, 0)
if args.cfips == 2:
    train, val = fe_cfips(train, val, 0, 1, 0)
if args.cfips == 3:
    train, val = fe_cfips(train, val, 0, 0, 1)
if args.cfips == 4:
    train, val = fe_cfips(train, val, 1, 1, 1)

if args.active > 0:
    train, val = fe_active(train, val)


# training
print('')
print('Start training with ', features)
# scaler = StandardScaler()
# scaler = MinMaxScaler()
xgb = XGBRegressor()
# xgb.fit(scaler.fit_transform(train[features]), train[target])
# prediction = xgb.predict(scaler.fit_transform(val[features]))
xgb.fit(train[features], train[target])
prediction = xgb.predict(val[features])

# eval
def SMAPE(a, f):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)

score = SMAPE(val[target].values.reshape(-1), prediction)
print('')
print('smape score: ', round(score, 4))



