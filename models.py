import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import torch
import torch.nn as nn



def train_xgb(train, val, features, target):
    print('')
    print('Model: ', 'xgbregressor')
    xgb = XGBRegressor()
    # xgb.fit(scaler.fit_transform(train[features]), train[target])
    xgb.fit(train[features], train[target])
    # prediction = xgb.predict(scaler.fit_transform(val[features]))
    prediction = xgb.predict(val[features])
    return prediction

def train_lgbm(train, val, features, target):
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
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
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


def train_dl(train_loader, test_loader, features, model_name, n_epochs=30):

    n_features = len(features)
    print('')
    if model_name == 'nn':
        print('model: nn.Linear')
        model = NN(n_features, hidden_size=64, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if model_name == 'lstm':
        print('model: LSTM')
        model = LSTM(n_features, hidden_size=64, num_layers=1, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if model_name == 'gru':
        print('model: GRU')
        model = GRU(n_features, hidden_size=128, num_layers=1, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

        if (n_epochs > 9) and (epoch%5 == 4):
            print(f"[{epoch+1}/{n_epochs}] Training loss: {training_loss:.4f}")
        if n_epochs < 10:
            print(f"[{epoch+1}/{n_epochs}] Training loss: {training_loss:.4f}")

    with torch.no_grad():
        predictions = []
        values = []
        for x_test, y_test in test_loader:
            model.eval()
            outputs = model(x_test)
            predictions.append(outputs)
            values.append(y_test)

    return predictions, values

