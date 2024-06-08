import concurrent
import concurrent.futures
import datetime
import logging
import os
import threading

import numpy as np
import pandas as pd
from absl import flags, app
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data', 'Directory that has historical stock data')
flags.DEFINE_string('output_dir', './output', 'Directory where research results should be saved')

_lock: threading.Lock = threading.Lock()
results = {}
param_grid_lr = {
    'fit_intercept': [True, False],
}

param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 1.0]
}

# Traditional ML models and their params
tr_model_and_params = [
    ('LinearRegression', LinearRegression(), param_grid_lr),
    ('DecisionTreeRegressor', DecisionTreeRegressor(), param_grid_dt),
    ('RandomForestRegressor', RandomForestRegressor(), param_grid_rf),
    ('XGBRegressor', XGBRegressor(), param_grid_xgb),
]


def _update_results(ticker: str, eval_result: dict):
    global results, _lock
    with _lock:
        results[ticker] = eval_result


def train_predict_evaluate_all(data_dir: str):
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for filename in os.listdir(data_dir):
            logging.info(f'processing {filename}')
            futures.append(executor.submit(_train_predict_evaluate, filename, data_dir))
    concurrent.futures.wait(futures)


def _train_predict_evaluate(filename: str, data_dir: str):
    ticker = filename.split('_')[0]
    raw_data = _load_historical_data(os.path.join(data_dir, filename))
    X_scaled, y = _preprocess_data(raw_data)
    logging.info(f'{ticker}: starting evaluation of all models')
    eval_data = _train_and_evaluate_tr_models(ticker, X_scaled, y)
    logging.info(f'{ticker}: completed evaluation of all models')
    _update_results(ticker, eval_data)


def _load_historical_data(filepath: str) -> pd.DataFrame:
    logging.info(f'loading historical data from {filepath}')
    raw_data = pd.read_csv(filepath)
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])
    raw_data.set_index('Date', inplace=True)
    # add a target column with % chg value from the next day
    raw_data['Target'] = (raw_data['Close'].shift(-1) - raw_data['Close']) / raw_data['Close']
    return raw_data


def _preprocess_data(raw_data: pd.DataFrame):
    raw_data.dropna(inplace=True)
    X = raw_data.drop(columns=['Target'])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y = raw_data['Target']
    return X_scaled, y


def _train_and_evaluate_tr_models(ticker, X_scaled, y):
    evaluation_results = {}
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    for name, model, param_grid in tr_model_and_params:
        logging.info(f'{ticker}: finding best estimator for {name}')
        estimator = _fit_and_get_best_estimator(model, param_grid, X_train, y_train)
        rmse = _evaluate_tr_model(estimator, X_test, y_test)
        evaluation_results[name] = rmse
    return evaluation_results


def _save_results():
    global results
    filename = os.path.join(FLAGS.output_dir, f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_results.csv')
    df: pd.DataFrame = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'Symbol'
    df.to_csv(filename)
    logging.info(f'results saved to {filename}')


def _fit_and_get_best_estimator(model, param_grid, X_train, y_train, n_jobs=-1, cv=5):
    searcher = GridSearchCV(estimator=model, param_grid=param_grid,
                            n_jobs=n_jobs, cv=cv, scoring='neg_mean_squared_error')
    searcher.fit(X_train, y_train)
    return searcher.best_estimator_


def _evaluate_tr_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse


def main(_):
    if not os.path.isdir(FLAGS.output_dir):
        logging.info(f'creating output dir {FLAGS.output_dir}')
        os.mkdir(FLAGS.output_dir)
    train_predict_evaluate_all(FLAGS.data_dir)
    _save_results()


if __name__ == '__main__':
    app.run(main)
