import yfinance as yf
import pandas as pd

ticker_symbol = 'EURUSD=X'
eurusd_data = yf.download(ticker_symbol, start='2025-01-01', end='2025-09-30')
eurusd_data.to_csv('eurusd_historical_data.csv', index=True)