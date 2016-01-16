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
import tabulate
import numpy as np
import pandas as pd
import StringIO

try:
    import cPickle as pickle
except:
    import pickle

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

def log_message(msg, log_location):
    """Display and log message"""
    # Display on stdout
    sys.stdout.write(msg)
    sys.stdout.flush()
    # Log to file
    for location in log_location:
        with open(location, 'a') as f:
            f.write(msg)

def update_history(
    symbol_manifest_location='./data/symbols.csv',
    history_status_location='./data/history.json',
    log_location='./data/log.txt',
    history_path='./data/history/{}'
):
    """
    Checks the current history in storage and downloads updates for any incomplete symbol.

    Inputs:
        symbol_manifest_location: file path string to store `.csv` manifest of symbols
        history_status_location: file path string to store `.json` status of current history
        log_location: file path string to store `.txt` log of function Operations
        history_path: directory path template string (e.g. `./path/to/history/{}`) to store `.pkl` of each symbol
        download_offset: the numbers of years to download at a time, defaults to 5

        Note: symbol_manifest_location, history_status_location, and log_location can also be passed lists of locations to have
        outputs written to multiple places, however this "master" location will only be read from first item in list (`[0]`).

    Symbol list is created from NASDAQ's list:
        ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt
        ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt

    Symbol History is requested in yearly chucks from Google's unofficial financial API:
        http://www.google.com/finance/historical?q={symbol}&startdate={start}&enddate={end}&output=csv

    If no history is found, the function is self contained and will generate it's directory structure and begin
    downloading all data since 1978-01-01.

    The function is ment to be an "always on" function that is called over and over running in the background either
    as something in a script that the user always runs or a cron job.

    This needs to be run at least "the number of symbols" times a day once history has been established,
    or set up to check/run every x seconds. Running every x seconds is recommended as the function handles checking
    before updating. With this way, the "download all history" feature will happen automagically every x seconds
    until full history is download, after which new history will only be downloaded once per symbol each a day,
    pausing until the next day.

    Anecdotally, Google did not like getting constant requests for intervals under 5 seconds. YMMV. It is also suggested to
    make the requests slightly random instead of exactly x seconds.  Successful downloads were acheived with a random internals
    between 1 and 5 seconds.

    """

    # Ensure list type
    symbol_manifest_location = [symbol_manifest_location] if isinstance(symbol_manifest_location, str) else symbol_manifest_location
    history_status_location = [history_status_location] if isinstance(history_status_location, str) else history_status_location
    log_location = [log_location] if isinstance(log_location, str) else log_location

    # Done for the day flag
    done = False

    # Times
    now = datetime.datetime.now()
    today = now.date()
    earliest_date = datetime.date(1900, 1, 1)
    download_offset = 1

    # Check if history directory exists
    if not os.path.exists(os.path.dirname(history_path)):
        os.makedirs(os.path.dirname(history_path))
        log_message('History directory does not exists. Creating Directory.\n', log_location)

    # Read in History Status
    if os.path.exists(history_status_location[0]):
        with open(history_status_location[0], 'r') as f:
            history_status = json.load(f)
            history_status['day'] = datetime.datetime.strptime(history_status['day'], '%Y-%m-%d').date()
            history_status['last'] = datetime.datetime.strptime(history_status['last'], '%Y-%m-%dT%H:%M:%S.%f')
            history_status['complete'] = False if history_status['day'] < today else True

    # New History Generation
    else:
        log_message('History Status does not exists. Creating Status.\n', log_location)
        history_status = {
            'count': 0,
            'complete': False,
            'last': datetime.datetime.now().isoformat(),
            'day': str(datetime.date.today()),
            'mode': 'build',
            'manifest': False,
            'symbol': None,
            'date': None,
            'number_of_symbols': 0,
            'downloaded': 0,
            'download_attempt': 0,
            'percent_complete': 0.0,
            'percent_attempt': 0.0
        }

    # If symbol manifest exist enter update mode
    if os.path.exists(symbol_manifest_location[0]):
        # Read from disk
        symbol_manifest = pd.read_csv(symbol_manifest_location[0], index_col=0, parse_dates=[6, 7])
        # If symbol history is not complete, incrementally download history backwards
        incomplete_symbols = symbol_manifest[~symbol_manifest['Current']]

        if len(incomplete_symbols) > 0:
            # Get first incomplete symbol
            symbol = incomplete_symbols.index.tolist()[0]
            # Set end date to today if first download or set to last start
            end = today if pd.isnull(symbol_manifest.loc[symbol]['End']) else symbol_manifest.loc[symbol]['Start'].date()
            # Set new start date to a year before end
            start = (end + pd.DateOffset(years=-download_offset)).date()
            # Clip end to earliest_date if start is before it
            start = earliest_date if start < earliest_date else start

            log_message('{:%Y-%m-%d %H:%M:%S}: Downloading {} from {} to {}:'.format(now, symbol, start, end), log_location)

            # Download data
            data = download_google_history(symbol, start, end)
            # Stop backward download if data empty
            if data.empty:
                symbol_manifest.loc[symbol, 'Current'] = True
            # If that date range returned data
            else:
                # If no end recorded, this is the first data returned, record end and store data
                if pd.isnull(symbol_manifest.loc[symbol]['End']):
                    symbol_manifest.loc[symbol, 'End'] = data.index[-1].date()
                    with open(history_path.format(symbol + '.pkl'), 'w') as f:
                        pickle.dump(data, f, protocol=2)
                else:
                    # Get current data, append new data, and write to disk
                    with open(history_path.format(symbol + '.pkl'), 'r') as f:
                        history = pickle.load(f)
                    history = history.append(data).sort_index()
                    with open(history_path.format(symbol + '.pkl'), 'w') as f:
                        pickle.dump(history, f, protocol=2)
                # Record start in manifest
                symbol_manifest.loc[symbol, 'Start'] = data.index[0].date()

            # Record in status
            history_status['symbol'] = symbol
            history_status['date'] = str(start)

            # Store manifest to disk
            for location in symbol_manifest_location:
                symbol_manifest.to_csv(location)
            log_message(' {}\n'.format('[ ]' if data.empty else '[x]'), log_location)

        else:
            log_message('No Incomplete Symbols. Shut down for the rest of the day.\n', log_location)
            history_status['last'] = True
            done = True

    # If symbol manifest doesn't exist begin to generate history
    else:
        log_message('Symbol Manifest does not exist. It will now be generated.\n', log_location)
        # Get DataFrame of symbols from nasdaq
        symbol_manifest = download_all_symbols()
        # Initialize data to track of
        symbol_manifest['Start'] = None
        symbol_manifest['End'] = None
        symbol_manifest['Current'] = False
        # Store to disk
        for location in symbol_manifest_location:
            symbol_manifest.to_csv(location)
        # Status
        history_status['manifest'] = True

    # Store status to disk at the end of the script
    history_status['last'] = datetime.datetime.now().isoformat()
    history_status['day'] = str(datetime.date.today())
    history_status['count'] += 1

    history_status['number_of_symbols'] = len(symbol_manifest)
    history_status['downloaded'] = symbol_manifest['Start'].count()
    history_status['download_attempt'] = symbol_manifest['End'].count()
    history_status['percent_complete'] = np.round(100.0 * symbol_manifest['Start'].count() / float(len(symbol_manifest)),2)
    history_status['percent_attempt'] = np.round(100.0 * symbol_manifest['End'].count() / float(len(symbol_manifest)), 2)

    for location in history_status_location:
        with open(location, 'w') as f:
            json.dump(history_status, f, indent=4, separators=(',',': '))

    for location in [h.replace('.json', '.txt') for h in history_status_location]:
        with open(location, 'w') as f:
            f.write(tabulate.tabulate(history_status.items()).replace(' ', '.'))

    return done
