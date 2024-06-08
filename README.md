## Spock - Stock predictions via ML; live long and prosper 🖖🏻

Provides models (traditional and neural network based) for stock predictions.

> NOTE: The `data_downloader.py` is hardcoded for NSE stocks, but can
> easily be changed to any exchange with minimal effort.

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
