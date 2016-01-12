"""
market.py

Operations that work on the whole market, either an index asset or a DataFrame of individual assets.
"""

import os
import sys
import json
import time
import urllib
import datetime
import numpy as np
import pandas as pd
import StringIO

# Download constants
URLPATTERN = 'http://www.google.com/finance/historical?q={symbol}&startdate={start}&enddate={end}&output=csv'
EXCHANGES = {'', 'NYSE:', 'NASDAQ:', 'NYSEMKT:', 'NYSEARCA:'}
NASDAQ_URL = 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/'
NASDAQ_FILE = 'nasdaqlisted.txt'
OTHERS_FILE = 'otherlisted.txt'
COLUMN_ORDER = ['Symbol', 'Security Name', 'Exchange', 'ETF', 'NASDAQ Symbol', 'Test Issue']
EXCHANGE_ABBR = {
    'Q' : 'NASDAQ',
    'A' : 'NYSE MKT',
    'N' : 'NYSE',
    'P' : 'ARCA',
    'Z' : 'BATS'
}

def download_all_symbols():
    """Download current symbols from NASDAQ server, return as DataFrame"""

    # Get NASDAQ symbols
    nasdaq_text = urllib.urlopen(NASDAQ_URL + NASDAQ_FILE).read()
    # Process NASDAQ symbols
    nasdaq = pd.read_csv(StringIO.StringIO(nasdaq_text), delimiter='|')
    # Drop Unneccesary data
    nasdaq = nasdaq.ix[:, :-1]
    # Set Exchange and ETFness
    nasdaq['ETF']= 'N'
    nasdaq['Exchange']= 'Q'
    # Clean Columns
    nasdaq['NASDAQ Symbol'] = nasdaq['Symbol']

    # Get OTHER (NYSE, BATS) symbols
    other_text = urllib.urlopen(NASDAQ_URL + OTHERS_FILE).read()
    # Process OTHER symbols
    other = pd.read_csv(StringIO.StringIO(other_text), delimiter='|')
    # Drop Unneccesary data
    other = other.ix[:, :-1]
    # Clean Columns
    other = other.rename(columns={'ACT Symbol': 'Symbol'})

    # Concatenate NASDAQ and OTHER data frames together
    symbols = pd.concat([nasdaq, other], ignore_index=False)
    symbols = symbols.sort_values(by='Symbol').reset_index(drop=True)
    symbols['Exchange'] = symbols['Exchange'].map(EXCHANGE_ABBR)
    symbols = symbols[COLUMN_ORDER]
    symbols = symbols.set_index('Symbol')

    return symbols

def download_google_history(symbols, start, end=datetime.date.today()) :
    """
    Download daily symbol history from Google servers for specified range
Returns DataFrame with Date, Open, Close, Low, High, Volume
    """

    # Set up empty DataFrame
    history = pd.DataFrame({'Open':[],'Close':[],'High':[],'Low':[],'Volume':[]})
    history.index.name = 'Date'

    # Check each exchange, bounce out once found
    for exchange in EXCHANGES:
        url_vars = {
            'symbol': exchange + symbols,
            'start' : start.strftime('%b %d, %Y'),
            'end' : end.strftime('%b %d, %Y')
        }
        google_url = URLPATTERN.format(**url_vars)
        google_string = urllib.urlopen(google_url).read()
        if (google_string.find('Not Found') < 0) :
            data = pd.read_csv(StringIO.StringIO(google_string), index_col=0, na_values=['','-'], parse_dates=True).sort_index()
            if len(data.index) > 0  and data.index[0].year == start.year:
                history = data
            break

    return history
