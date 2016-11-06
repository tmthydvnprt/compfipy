"""
market.py

Operations that work on the whole market, either an index asset or a DataFrame of individual assets.
"""

import os
import sys
import json
import time
import urllib2
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
GOOGLE_URL = 'http://www.google.com/finance/historical?q={symbol}&startdate={start}&enddate={end}&output=csv'
YAHOO_URL = 'http://ichart.finance.yahoo.com/table.csv?s={symbol}&c={start}'
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
    nasdaq_text = urllib2.urlopen(NASDAQ_URL + NASDAQ_FILE).read()
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
    other_text = urllib2.urlopen(NASDAQ_URL + OTHERS_FILE).read()
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
    history = pd.DataFrame({'Open':[], 'Close':[], 'High':[], 'Low':[], 'Volume':[]})
    history.index.name = 'Date'

    # Check each exchange, bounce out once found
    for exchange in EXCHANGES:
        url_vars = {
            'symbol': exchange + symbols,
            'start' : start.strftime('%b %d, %Y'),
            'end' : end.strftime('%b %d, %Y')
        }
        url = GOOGLE_URL.format(**url_vars)
        result_string = urllib2.urlopen(url).read()
        if (result_string.find('Not Found') < 0) :
            data = pd.read_csv(StringIO.StringIO(result_string), index_col=0, na_values=['','-'], parse_dates=True).sort_index()
            if len(data.index) > 0  and data.index[0].year == start.year:
                history = data
            break

    return history

def download_yahoo_history(symbols, start) :
    """
    Download daily symbol history from Yahoo servers for specified range
    Returns DataFrame with Date, Open, Close, Low, High, Volume
    """

    # Set up empty DataFrame
    history = pd.DataFrame({'Open':[],'Close':[],'High':[],'Low':[],'Volume':[]})
    history.index.name = 'Date'

    # Check each exchange, bounce out once found
    for exchange in EXCHANGES:
        url_vars = {
            'symbol': exchange + symbols,
            'start' : start.strftime('%Y-%m-%d')
        }
        url = YAHOO_URL.format(**url_vars)
        result_string = urllib2.urlopen(url).read()
        if (result_string.find('Not Found') < 0) :
            data = pd.read_csv(
                StringIO.StringIO(result_string),
                index_col=0,
                na_values=['','-'],
                parse_dates=True
            ).sort_index()
            if len(data.index) > 0:
                history = data
            break

    return history

def log_message(msg, log_location, log=True, display=True):
    """Display and log message"""
    # Display on stdout
    if display:
        sys.stdout.write(msg)
        sys.stdout.flush()
    # Log to file
    if log:
        for location in log_location:
            with open(location, 'a') as f:
                f.write(msg)

def update_history(
    symbol_manifest_location='./data/symbols.csv',
    history_status_location='./data/history.json',
    log_location='./data/log.txt',
    history_path='./data/history/{}',
    source='google',
    log=True,
    display=True
):
    """
    Checks the current history in storage and downloads updates for any incomplete symbol.

    Inputs:
        symbol_manifest_location : file path string to store `.csv` manifest of symbols
        history_status_location  : file path string to store `.json` status of current history
        log_location             : file path string to store `.txt` log of function Operations
        history_path             : directory path template string (e.g. `./path/to/history/{}`) to store `.pkl` of each symbol
        download_offset          : the numbers of years to download at a time, defaults to 5
        source                   : string identifying data source, can be `'yahoo'` or `'google'`, defaults to `'google'`
        log                      : boolean to toggle update process being logged to a file, defaults to True
        display                  : boolean to toggle update process being displayed on stdout, defaults to True

        Note: symbol_manifest_location, history_status_location, and log_location can also be passed lists of locations to have
        outputs written to multiple places, however this "master" location will only be read from first item in list (`[0]`).

    Symbol list is created from NASDAQ's list:
        ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt
        ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt

    Symbol History is requested in yearly chucks from Google's unofficial financial API:
        http://www.google.com/finance/historical?q={symbol}&startdate={start}&enddate={end}&output=csv

    or from Yahoo's unofficial financial API:
        http://ichart.finance.yahoo.com/table.csv?s={symbol}&c={start}

    If no existing history is found on disk, the function is self contained and will generate the directory structure and begin
    downloading all data since 1977-01-01.

    The function is ment to be an "always on" function that is called over and over, running in the background either
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
    earliest_date = datetime.date(1977, 1, 1)
    download_offset = 1

    # Check if data directory exists
    for location in symbol_manifest_location:
        if not os.path.exists(os.path.dirname(location)):
            os.makedirs(os.path.dirname(location))
            log_message(
                'Data directory does not exists. Creating Directory.\n{}'.format(os.path.dirname(location)),
                log_location,
                log,
                display
            )

    # Check if history directory exists
    if not os.path.exists(os.path.dirname(history_path)):
        os.makedirs(os.path.dirname(history_path))
        log_message(
            'History directory does not exists. Creating Directory.\n{}'.format(os.path.dirname(history_path)),
            log_location,
            log,
            display
        )

    # Read in History Status
    if os.path.exists(history_status_location[0]):
        with open(history_status_location[0], 'r') as f:
            history_status = json.load(f)
            history_status['day'] = datetime.datetime.strptime(history_status['day'], '%Y-%m-%d').date()
            history_status['last'] = datetime.datetime.strptime(history_status['last'], '%Y-%m-%dT%H:%M:%S.%f')
            history_status['complete'] = False if history_status['day'] < today else True

    # New History Generation
    else:
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
        log_message('History Status does not exists. Creating Status.\n', log_location, log, display)

    # If symbol manifest exist enter update mode
    if os.path.exists(symbol_manifest_location[0]):
        # Read from disk
        symbol_manifest = pd.read_csv(symbol_manifest_location[0], index_col=0, parse_dates=[6, 7])

        # If past symbol history is not complete, incrementally download history backwards
        incomplete_history = symbol_manifest[~symbol_manifest['Current']]
        if len(incomplete_history) > 0:
            # Get first incomplete symbol
            symbol = incomplete_history.index.tolist()[0]

            if source is 'yahoo':
                # Set end date to today
                end = today
                # Set new start date to a year before end
                start = earliest_date
            else:
                # Set end date to today if first download or set to last start
                end = today if pd.isnull(symbol_manifest.loc[symbol]['End']) else symbol_manifest.loc[symbol]['Start'].date()
                # Set new start date to a year before end
                start = (end + pd.DateOffset(years=-download_offset)).date()
                # Clip end to earliest_date if start is before it
                start = earliest_date if start < earliest_date else start
            log_message(
                '{:%Y-%m-%d %H:%M:%S}: Downloading {} from {} to {}:'.format(now, symbol, start, end),
                log_location,
                log,
                display
            )

            # Download data
            if source is 'yahoo':
                data = download_yahoo_history(symbol, start)
            else:
                data = download_google_history(symbol, start, end)

            if source is 'yahoo':
                # Stop backward download because all years occur at once
                symbol_manifest.loc[symbol, 'Current'] = True
            elif source is 'google' and data.empty:
                # Stop backward download because data is empty
                symbol_manifest.loc[symbol, 'Current'] = True

            # If that date range returned data
            if not data.empty:
                # If no end recorded, this is the first data returned, record end and store data
                if pd.isnull(symbol_manifest.loc[symbol]['End']):
                    symbol_manifest.loc[symbol, 'End'] = data.index[-1].date()
                    with open(history_path.format(symbol + '.pkl'), 'w') as f:
                        pickle.dump(data, f, protocol=2)
                else:
                    # Get current data
                    with open(history_path.format(symbol + '.pkl'), 'r') as f:
                        history = pickle.load(f)
                    # Append data
                    history = history.append(data).sort_index()
                    # Make sure dupicate dates are removed
                    history = history[~history.index.duplicated(keep='first')]
                    # Write to disk
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
            log_message(' {}\n'.format('[ ]' if data.empty else '[x]'), log_location, log, display)

        else:
            # If current symbol history is not complete, incrementally download history forwards
            incomplete_symbols = symbol_manifest.loc[
                (symbol_manifest['End'] != today) &
                ~pd.isnull(symbol_manifest['Start'])
            ]
            if len(incomplete_symbols) > 0:
                # Get first incomplete symbol
                symbol = incomplete_symbols.index.tolist()[0]
                # Get new start date as last end date
                start = symbol_manifest.loc[symbol]['End'].date() + pd.DateOffset(days=1)
                # Set new end date to a year from start
                end = (start + pd.DateOffset(years=download_offset)).date()
                # Clip end to today if end is in the future
                end = today if end > today else end
                log_message(
                    '{:%Y-%m-%d %H:%M:%S}: Downloading {} from {} to {}:'.format(now, symbol, start, end),
                    log_location,
                    log,
                    display
                )

                # Download data
                if source is 'yahoo':
                    data = download_yahoo_history(symbol, start)
                else:
                    data = download_google_history(symbol, start, end)

                # If that date range returned data
                if not data.empty:
                    # Get current data
                    with open(history_path.format(symbol + '.pkl'), 'r') as f:
                        history = pickle.load(f)
                    # Append data
                    history = history.append(data).sort_index()
                    # Make sure dupicate dates are removed
                    history = history[~history.index.duplicated(keep='first')]
                    # Write to disk
                    with open(history_path.format(symbol + '.pkl'), 'w') as f:
                        pickle.dump(history, f, protocol=2)

                # Record last ending in manifest
                symbol_manifest.loc[symbol, 'End'] = end

                # Record in status
                history_status['symbol'] = symbol
                history_status['date'] = str(start)

                # Store manifest to disk
                for location in symbol_manifest_location:
                    symbol_manifest.to_csv(location)
                log_message(' {}\n'.format('[ ]' if data.empty else '[x]'), log_location, log, display)

            else:
                log_message('No Incomplete Symbols. Shut down for the rest of the day.\n', log_location, log, display)
                history_status['last'] = True
                done = True

    # If symbol manifest doesn't exist begin to generate history
    else:
        log_message('Symbol Manifest does not exist. It will now be generated.\n', log_location, log, display)
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
