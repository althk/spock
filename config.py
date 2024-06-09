from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

param_grid_lr = {
    'fit_intercept': [True, False],
}

param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 8, 10, 13, 21],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8]
}

param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 8, 13, 21],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0.1, 1, 10],
}

# Traditional ML models and their params
tr_model_cfg = {
    'LinearRegression': (LinearRegression(), param_grid_lr),
    'DecisionTreeRegressor': (DecisionTreeRegressor(), param_grid_dt),
    'RandomForestRegressor': (RandomForestRegressor(), param_grid_rf),
    'XGBRegressor': (XGBRegressor(), param_grid_xgb),
}
