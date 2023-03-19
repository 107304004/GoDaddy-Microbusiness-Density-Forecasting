import numpy as np
import pandas as pd

from feature_engineering import fe_active, fe_cfips, fe_shift
from sklearn.preprocessing import MinMaxScaler
# import models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import torch
import torch.nn as nn
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
    data, features = fe_shift(data, features, 12)

train, val = cut_valid(data, args.cutval_date)

if args.cfips > 0:
    print('Preparing cfips feature...')
    train, val, features = fe_cfips(train, val, features, 1, 1, 1)

if args.active > 0:
    print('Preparing new active & adult_pop...')
    train, val, features = fe_active(train, val, features)


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

class NN(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # forward propagate LSTM
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        # unsqueeze: (64,n_features) -> (64, 1, n_features)
        # out_shape: (batch_size, seq_length, hidden_size)

        # decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # forward propagate
        out, _ = self.gru(x.unsqueeze(1), h0)

        # decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# deep learning training settings
batch_size = 64

# scaler df to tensor
train_for_dl, val_for_dl = cut_valid(train, '2022-11')
scaler = MinMaxScaler()
train_for_dl_fe = scaler.fit_transform(train_for_dl[features])
val_for_dl_fe = scaler.fit_transform(val_for_dl[features])
test_for_dl_fe = scaler.fit_transform(val[features])
x, y = torch.Tensor(train_for_dl_fe), torch.Tensor(train_for_dl[target].values)
val_x, val_y = torch.Tensor(val_for_dl_fe), torch.Tensor(val_for_dl[target].values)
test_x, test_y = torch.Tensor(test_for_dl_fe), torch.Tensor(val[target].values)

# Dataset
train_dataset = TensorDataset(x, y)
val_dataset = TensorDataset(val_x, val_y)
test_dataset = TensorDataset(test_x, test_y)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

n_features = len(features)

def train_dl(model_name, n_epochs=30):

    print('')
    if model_name == 'nn':
        print('model: nn.Linear')
        model = NN(n_features, hidden_size=64, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    if model_name == 'lstm':
        print('model: LSTM')
        model = LSTM(n_features, hidden_size=64, num_layers=1, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
    if model_name == 'gru':
        print('model: GRU')
        model = GRU(n_features, hidden_size=64, num_layers=1, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)

    # n_epochs = 6
    for epoch in range(n_epochs):

        batch_losses = []
        for x_batch, y_batch in train_loader:
            # forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            # print(outputs.shape, y_batch.shape, loss.item())
        training_loss = np.mean(batch_losses)

        with torch.no_grad():
            batch_val_losses = []
            for x_val, y_val in val_loader:
                model.eval()
                outputs = model(x_val)
                val_loss = criterion(outputs, y_val)
                batch_val_losses.append(val_loss)
            validation_loss = np.mean(batch_val_losses)
        if (n_epochs > 9) and (epoch%5 == 4):
            print(f"[{epoch+1}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")
        if n_epochs < 10:
            print(f"[{epoch+1}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")

    with torch.no_grad():
        predictions = []
        values = []
        for x_test, y_test in test_loader:
            model.eval()
            outputs = model(x_test)
            predictions.append(outputs)
            values.append(y_test)

    return predictions, values


# eval
def SMAPE(a, f):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)

score = SMAPE(val[target].values.reshape(-1), train_xgb())
print('smape score: ', round(score, 4))

score = SMAPE(val[target].values.reshape(-1), train_lgbm())
print('smape score: ', round(score, 4))

p, v = train_dl('nn', n_epochs=6)
p = torch.cat([t.view(-1) for t in p]).numpy()
v = torch.cat([t.view(-1) for t in v]).numpy()
print('smape score: ', round(SMAPE(p, v), 5))

p, v = train_dl('lstm')
p = torch.cat([t.view(-1) for t in p]).numpy()
v = torch.cat([t.view(-1) for t in v]).numpy()
print('smape score: ', round(SMAPE(p, v), 5))

p, v = train_dl('gru')
p = torch.cat([t.view(-1) for t in p]).numpy()
v = torch.cat([t.view(-1) for t in v]).numpy()
print('smape score: ', round(SMAPE(p, v), 5))


