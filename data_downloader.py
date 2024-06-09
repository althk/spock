"""Downloads historical stock data from Yahoo! Finance and adds some TA indicators.

It does the following tasks:
1. Downloads historical data for the given list of stocks.
2. Add TA indicators.
3. Saves the historical data along with the TA indicators to disk.

Details of each task are given below.

1. Downloading data
- The symbols should be provided in a csv file via the flag --symbols_file
- The format of the csv file should be: "Stock name","Stock symbol". Example:
"State Bank of India","SBIN"
"Reliance Industries","RELIANCE"
- The symbols must be from the NSE exchange.
- An existing symbols file is provided for NFO symbols (NSE FnO stocks).

2. Adding TA indicators
- After OHLCV data is downloaded, the script adds the following indicators to the data:
- SMA_10, SMA-50, SMA-200, RSI_14, RSI_SMA_14, VSTOP_10_2

3. Saves the data along with the TA indicators to disk
- The data is downloaded to the directory specified in the flag --data_dir.
- Each symbol is stored in a separate csv file in the data directory.
- If the data directory does not exist, it will be created.
- If an existing data file exists for a symbol, it will be overwritten.
- The file name is `Symbol_startdate_enddate_interval.csv`, e.g. SBIN.NS_20220401_20230331_1h.csv
"""
import csv
import datetime
import os
from typing import Dict

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from absl import flags, app, logging

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', './data',
                    'Directory where historical data should be downloaded to.')
flags.DEFINE_string('symbols_file', './nfo_symbols.csv',
                    'name of the CSV file containing stock symbols.'
                    'The format must be "stock name","stock symbol"')
flags.DEFINE_string('start_date', None, 'Start date in YYYY-MM-DD format. '
                                        'Defaults to yesterday - 2 years if not provided.')
flags.DEFINE_string('end_date', None, 'End date in YYYY-MM-DD format. '
                                      'Defaults to yesterday if not provided.')


def load_symbols(filename: str) -> Dict[str, str]:
    nfo_dict = {}
    with open(filename) as fp:
        reader = csv.reader(fp)
        for row in reader:
            nfo_dict[row[1]] = row[0]
    return nfo_dict


def _dl_data(symbols: str, data_dir: str, start_date: datetime.date = None,
             end_date: datetime.date = None, interval='1d') -> None:
    data = yf.download(symbols, start=start_date, end=end_date,
                       interval=interval, threads=10, group_by='ticker')
    tickers = [s.strip() for s in symbols.split(' ')]
    for ticker in tickers:
        t: pd.DataFrame = data[ticker]
        df: pd.DataFrame = t.drop(columns=['Adj Close'])
        logging.debug(f'Adding TA data to {ticker}')
        _add_ta(df)
        filename = os.path.join(
            data_dir,
            f'{ticker}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}_{interval}.csv')
        df.to_csv(filename)
        logging.info(f'Saved {ticker} data to {filename}')


def _add_ta(df: pd.DataFrame):
    df['SMA_10'] = df.ta.sma(10)
    df['SMA_50'] = df.ta.sma(50)
    df['SMA_200'] = df.ta.sma(200)
    df['RSI_14'] = df.ta.rsi(14)
    df['RSI_SMA_14'] = ta.rsi(df['RSI_14'], 14)
    df['VSTOP_10_1'] = _calc_vstop(df, 10, 1.0)


def _calc_vstop(df: pd.DataFrame, period: int, multiplier: float) -> pd.Series:
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=period)
    long_stop = df['High'] - (atr * multiplier)
    short_stop = df['Low'] + (atr * multiplier)
    vstop = pd.Series(np.where(df['Close'] > df['Close'].shift(1), long_stop, short_stop), index=df.index)
    return vstop


def download_historical_data(symbol_dict: Dict[str, str], data_dir: str, start_date, end_date) -> None:
    symbols = ['%s.NS' % k for k in symbol_dict.keys()]
    batch_size = 50
    completed = 0
    while completed < len(symbols):
        size = min(batch_size, len(symbols) - completed)
        s_str = ' '.join(symbols[completed:completed + size])
        _dl_data(s_str, data_dir, start_date, end_date)
        completed += size


def main(_):
    logging.info('Loading symbols from %s', FLAGS.symbols_file)
    symbol_dict = load_symbols(FLAGS.symbols_file)
    logging.info('Loaded %d symbols', len(symbol_dict))
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    end_date = datetime.date.today() - datetime.timedelta(days=1)  # yesterday

    if FLAGS.end_date:
        end_date = datetime.date.fromisoformat(FLAGS.end_date)
    if FLAGS.start_date:
        start_date = datetime.date.fromisoformat(FLAGS.start_date)
    else:
        start_date = end_date - datetime.timedelta(days=365 * 2)
    logging.info(f'downloading historical data from {start_date.isoformat()} '
                 f'to {end_date.isoformat()}')
    download_historical_data(symbol_dict, FLAGS.data_dir, start_date, end_date)


if __name__ == '__main__':
    app.run(main=main)
