## GoDaddy - Microbusiness Density Forecasting
https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/overview

#### We create 3 different feature engineering and test 5 combination of them
- Baseline (use no feature engineering)
- Lags
- Lags + CFIPS
- Lags + Active
- Lags + CFIPS + Active

#### We try 5 different models including
- XGB
- LGBM
- Linear Regression
- LSTM
- GRU

#### To run the exp
```bash
./run_0316.sh
```

The exp run through 5 ways of feature engineering by using 5 different models to have 25 results that evaluated on SMAPE.

|  | Baseline | Lags | Lags+CFIPS | Lags+Active | ALL |
| :-----:| :----: | :----: | :----: | :----: | :----: |
| XGB | 26.5953 | 2.0921 | 2.2582 | 2.1184 | 2.2330 |
| LGBM | 32.3634 | 2.0666 | 2.2265 | 2.1007 | 2.2058 |
| LR | 57.1285 | 57.1252 | 59.3354 | 59.9489 | 60.1480 |
| LSTM | 46.3726 | 31.0529 | 16.8579 | 21.5334 | 20.2403 |
| GRU | 47.7848 | 21.7134 | 16.2483 | 8.9433 | 16.8651 |
