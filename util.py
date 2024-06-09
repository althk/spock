import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


def load_historical_data(filepath: str) -> pd.DataFrame:
    raw_data = pd.read_csv(filepath)
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])
    raw_data.set_index('Date', inplace=True)
    # add a target column with % chg value from the next day
    raw_data['Target'] = (raw_data['Close'].shift(-1) - raw_data['Close']) / raw_data['Close']
    return raw_data


def preprocess_data(raw_data: pd.DataFrame):
    raw_data.dropna(inplace=True)
    X = raw_data.drop(columns=['Target'])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y = raw_data['Target']
    return scaler, X_scaled, y


def evaluate_tr_model(model_inst, param_grid, X_train, y_train, X_test, y_test, n_jobs=-1, cv=5):
    """Returns the RMSE for the predictions on the given test data on the best estimator."""
    # find the best estimator using grid search cross validation
    searcher = GridSearchCV(estimator=model_inst, param_grid=param_grid,
                            n_jobs=n_jobs, cv=cv, scoring='neg_mean_squared_error')
    searcher.fit(X_train, y_train)
    estimator = searcher.best_estimator_

    # evaluate the best estimator on the test data
    predictions = estimator.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse
