#! /bin/sh

url='https://query1.finance.yahoo.com/v7/finance/download/DOGE-USD?period1=1636839159&period2=1668375159&interval=1d&events=history&includeAdjustedClose=true'
user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'

wget -U "$user_agent" "$url" -O ./doge-prediction/data/01_raw/doge-usd.csv