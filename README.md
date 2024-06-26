## Spock - Stock predictions via ML; live long and prosper 🖖🏻

> NOTE: The `data_downloader.py` is hardcoded for NSE stocks, but can
> easily be changed to any exchange with minimal effort.


Evaluates models (traditional and neural network based) for stock predictions to find the best fit.

The following models are currently included in the evaluation:

- Traditional models
    - LinearRegression
    - DecisionTreeRegressor
    - RandomForestRegressor
    - XGBoostRegressor

### High level overview

1. Download historical data
2. Train and evaluate each model on each stock's data
3. Find the overall best model across all stocks' evaluation
4. Train that model on each stock and save the model per stock
5. For real-time predictions, load the model instance for the requested ticker and make prediction

### Steps to get started

1. Install the pre-reqs:

```shell
$ pip install pandas pandas-ta scikit-learn xgboost  # for traditional ML models
$ pip install tensorflow   # for deep learning models
$ pip install absl-py
```

2. Download raw data (this project uses `yfinance`) using the `data_downloader.py` script:

```shell
$ python data_downloader.py --symbols_file=./nfo_symbols.csv --data_dir=./data --logtostderr
```

3. Run `prediction_research.py` which then loads the downloaded data, trains a few models, evaluates
   them and finally stores some information about which model scored the best (`lowest RMSE`)

```shell
$ python prediction_research.py --data_dir='./data' --output_dir='./output' --logtostderr
```

### Interpreting the results

1. The last step above dumps the evaluation results under the `output_dir/YYYYmmddHHMMSS/evaluation_results.csv`
2. Run `parse_evaluation_results.py` to find out the best overall model

```shell
$ python parse_evaluation_results.py --evaluation_results_file=<path to the csv>
```

This will print out some statistics and highlight the best model overall for all stocks
> NOTE: the script currently picks the model with the min RMSE at p95

### Training the selected/best model on all stocks

1. Now that we have found the model that works best overall across all stocks,
   we need to train it on all the stocks individually and save the model per stock. For example, if
the evaluation results were saved at `output/20240609185957/evaluation_results.csv` and the stock data is in
the `data` directory, running the following command would parse the model selected via grid search cv
for each stock and save the model and scaler objects under the same output directory.
```shell
$ python train_model.py --data_dir=data --results_file=output/20240609185957/evaluation_results.csv --logtostderr
```
2. For example, if the model selected was `XGBRegressor`, then in the dir `output/20240609185957` there will
be files of the format `{ticker}_XGBRegressor.pkl` and `{ticker}_Scaler.pkl`.

### Making predictions

To predict values for a given ticker:
1. Load the model and scaler for the specific ticker
2. Scale the data using the loaded scaler
3. Predict the price using the loaded model