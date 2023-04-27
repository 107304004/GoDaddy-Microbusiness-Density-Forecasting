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

The exp run through 5 ways of feature engineering by using 5 different models to have 25 results.

|  | Baseline | Lags | Lags+CFIPS | Lags+Active | ALL |
| :-----:| :----: | :----: | :----: | :----: | :----: |
| XGB | 单元格 | 单元格 | 单元格 | 单元格 | 单元格 |
| LGBM | 单元格 | 单元格 | 单元格 | 单元格 | 单元格 |
| LR | 单元格 | 单元格 | 单元格 | 单元格 | 单元格 |
| LSTM | 单元格 | 单元格 | 单元格 | 单元格 | 单元格 |
| GRU | 单元格 | 单元格 | 单元格 | 单元格 | 单元格 |
