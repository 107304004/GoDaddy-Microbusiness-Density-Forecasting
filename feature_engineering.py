import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fe_shift(df, features, lag=1):

    for i in range(lag):
        df[f'mbd_lag_{i+1}'] = df.groupby('cfips')['microbusiness_density'].shift(i+1)
        features.append(f'mbd_lag_{i+1}')

    return df.loc[df[f'mbd_lag_{lag}'].notnull()], features


def fe_cfips(train, test, features, mean=0, std=0, trend=0):

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

    return train, test, features


def fe_active(train, test, features):

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

    return train, test, features



