import concurrent
import concurrent.futures
import datetime
import logging
import os
import threading

import pandas as pd
from absl import flags, app
from sklearn.model_selection import train_test_split

import config
import util

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data', 'Directory that has historical stock data')
flags.DEFINE_string('output_dir', './output', 'Directory where research results should be saved')

_lock: threading.Lock = threading.Lock()
results = {}


def _update_results(ticker: str, eval_result: dict):
    global results, _lock
    with _lock:
        results[ticker] = eval_result


def train_predict_evaluate_all(data_dir: str):
    """Evaluates all models for all stocks."""
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        data_files = os.listdir(data_dir)
        data_files.sort()
        total = len(data_files)
        for i, filename in enumerate(data_files):
            logging.info(f'({i + 1}/{total}) processing {filename}')
            futures.append(executor.submit(_train_predict_evaluate, filename, data_dir))
    concurrent.futures.wait(futures)


def _train_predict_evaluate(filename: str, data_dir: str):
    """Evaluates all models per individual stock identified by the filename."""
    ticker = filename.split('_')[0]
    filepath = os.path.join(data_dir, filename)
    logging.info(f'{ticker}: loading data from {filepath}')
    raw_data = util.load_historical_data(filepath)
    scaler, X_scaled, y = util.preprocess_data(raw_data)
    logging.info(f'{ticker}: starting evaluation of all models')
    eval_data = _train_and_evaluate_tr_models(ticker, X_scaled, y)
    logging.info(f'{ticker}: completed evaluation of all models')
    _update_results(ticker, eval_data)


def _train_and_evaluate_tr_models(ticker, X_scaled, y):
    """Evaluates configured traditional models and returns a dict of model name, RMSE."""
    evaluation_results = {}
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    for model_name, cfg in config.tr_model_cfg.items():
        logging.info(f'{ticker}: finding best estimator for {model_name}')
        model_factory, param_grid = cfg
        _, rmse = util.evaluate_tr_model(model_factory(), param_grid, X_train, y_train, X_test, y_test)
        evaluation_results[model_name] = rmse
    return evaluation_results


def _save_final_eval_results(eval_results_filename: str):
    """Saves the combined eval results for all tickers across all models to disk."""
    global results
    logging.info(f'saving final evaluation results to {eval_results_filename}')
    df: pd.DataFrame = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'Symbol'
    df.sort_index(inplace=True)
    df.to_csv(eval_results_filename)
    logging.info(f'evaluation results saved to {eval_results_filename}')


def main(_):
    if not os.path.isdir(FLAGS.output_dir):
        logging.info(f'creating output dir {FLAGS.output_dir}')
        os.mkdir(FLAGS.output_dir)
    eval_results_dir = os.path.join(FLAGS.output_dir, 'evaluation_results')
    os.makedirs(eval_results_dir, exist_ok=True)
    eval_results_filename = os.path.join(eval_results_dir,
                                         f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    train_predict_evaluate_all(FLAGS.data_dir)
    _save_final_eval_results(eval_results_filename)


if __name__ == '__main__':
    app.run(main)
