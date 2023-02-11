import pandas_datareader.data as web
import datetime as dt
import yfinance as yfin

crypto_currency = 'BTC'
against_currency = 'USD'

start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

yfin.pdr_override()
data = web.DataReader(f'{crypto_currency}-{against_currency}', start, end)

data.to_csv("crypto_data.csv")


