import concurrent.futures
import os.path

import joblib
from absl import flags, logging, app
from sklearn.model_selection import train_test_split

import util

FLAGS = flags.FLAGS

flags.DEFINE_string('results_file', None,
                    'Path to the results file that should be parsed to get the best model.')
flags.DEFINE_string('data_dir', None,
                    'Path to the directory that contains the training data.')


def _load_and_train(model_name, data_dir, output_dir):
    data_files = os.listdir(data_dir)
    total = len(data_files)
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for i, data_file in enumerate(data_files):
            file_path = os.path.join(data_dir, data_file)
            ticker = data_file.split('_')[0]
            logging.info(f'{i+1}/{total} processing {file_path}')
            future = executor.submit(_train_and_save, model_name, file_path, ticker, output_dir)
            futures.append(future)
    concurrent.futures.wait(futures)


def _train_and_save(model_name, file_path, ticker, output_dir):
    logging.info(f'{ticker}: loading data from {file_path}')
    raw_data = util.load_historical_data(file_path)
    scaler, X_scaled, y = util.preprocess_data(raw_data)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    logging.info(f'{ticker}: X_train.shape = {X_train.shape}, X_test.shape = {X_test.shape}'
                 f'y_train.shape = {y_train.shape}, y_test.shape = {y_test.shape}')
    logging.info(f'{ticker}: running grid search cv for best estimator')
    model_factory, param_grid = util.get_model(model_name)
    estimator, _ = util.evaluate_tr_model(model_factory, param_grid, X_train, y_train, X_test, y_test)
    _save_model(estimator, scaler, output_dir, model_name, ticker)


def _save_model(estimator, scaler, output_dir, model_name, ticker):
    model_file, scaler_file = util.model_scaler_filenames(ticker, model_name)
    model_path = os.path.join(output_dir, model_file)
    scaler_path = os.path.join(output_dir, scaler_file)
    joblib.dump(estimator, model_path)
    joblib.dump(scaler, scaler_path)


def main(_):
    output_dir, _ = os.path.split(FLAGS.results_file)
    logging.info(f'parsing results from {FLAGS.results_file}')
    _, best_model_name, _ = util.parse_evaluation_results(FLAGS.results_file)
    logging.info(f'Best model: {best_model_name}')
    _load_and_train(best_model_name, FLAGS.data_dir, output_dir)


if __name__ == '__main__':
    app.run(main)
