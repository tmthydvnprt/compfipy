"""
market.py

Operations that work on the whole market, either an index asset or a DataFrame of individual assets.

Symbol Lists are downloaded from NASDAQ FTP, ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt and
ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt The file definition is located at
http://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs.

"""

import os
import sys
import json
import urllib
import urllib2
import datetime
import StringIO

import calendar as cal
import cPickle as pickle
import numpy as np
import pandas as pd

import dateutil.easter
import tabulate

# Download Constants
# ------------------------------------------------------------------------------------------------------------------------------
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

# Date Helper Functions
# ------------------------------------------------------------------------------------------------------------------------------
def next_open_day(date=datetime.date.today()):
    """
    Find the next date the NYSE is open.
    """
    # Add one day to current date
    date = date + datetime.timedelta(days=1)
    # Continue adding days until the market is open
    while not is_open_on(date):
        date = date + datetime.timedelta(days=1)
    return date

def move_weekend_holiday(d):
    """
    If the holiday is part of the weekend, move it to the appropriate day.
    """
    # Saturday, make holiday friday before
    if d.weekday() == 5:
        return d - datetime.timedelta(days=1)
    # Sunday, make holiday monday after
    elif d.weekday() == 6:
        return d + datetime.timedelta(days=1)
    else:
        return d

def nth_week_day_of_month(n, weekday, month=datetime.date.today().month, year=datetime.date.today().year):
    """
    Get the nth weekday of a month during the year.
    """

    if isinstance(weekday, str) and len(weekday) == 3:
        weekday = list(cal.day_abbr).index(weekday)
    elif isinstance(weekday, str) and len(weekday) > 3:
        weekday = list(cal.day_name).index(weekday)

    if n > 0:
        first_day_of_month = datetime.date(year, month, 1)
        weekday_difference = (weekday - first_day_of_month.weekday()) % 7
        first_weekday_of_month = first_day_of_month + datetime.timedelta(days=weekday_difference)
        return first_weekday_of_month + datetime.timedelta(days=(n - 1) * 7)
    else:
        last_day_of_month = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
        weekday_difference = (last_day_of_month.weekday() - weekday) % 7
        last_weekday_of_month = last_day_of_month - datetime.timedelta(days=weekday_difference)
        return last_weekday_of_month - datetime.timedelta(days=(abs(n) - 1) * 7)
        # return cal.Calendar(weekday).monthdatescalendar(year,month)[n][0]
        # reply on stackoverflow that this has bugs, didn't work for the third monday in feburary 2016

def nyse_holidays(year=datetime.date.today().year):
    """
    Calulate the holidays of the NYSE for the given year.
    """
    if year < 1817:
        print 'The NYSE was not open in ' + str(year) +'! It was founded in March 8, 1817. Returning empty list []'
        return []
    else:
        typical_holidays = [
            datetime.date(year, 1, 1),                                 # New Year's Day
            nth_week_day_of_month(3, 'Mon', 1, year),                  # Martin Luther King, Jr. Day
            nth_week_day_of_month(3, 'Mon', 2, year),                  # Washington's Birthday (President's Day)
            dateutil.easter.easter(year) - datetime.timedelta(days=2), # Good Friday
            nth_week_day_of_month(-1, 'Mon', 5, year),                 # Memorial Day
            datetime.date(year, 7, 4),                                 # Independence Day
            nth_week_day_of_month(1, 'Mon', 9, year),                  # Labor Day
            nth_week_day_of_month(4, 'Thu', 11, year),                 # Thanksgiving Day
            datetime.date(year, 12, 25)                                # Christmas Day
        ]
        historical_holidays = [
            datetime.date(2012, 10, 29), # hurricane sandy
            datetime.date(2012, 10, 30), # hurricane sandy
        ]
        # Grab historical holidays for the year
        special_holidays = [v for v in historical_holidays if v.year == year]

        # Alter weekend holidays and add special holidays
        holidays = [move_weekend_holiday(h) for h in typical_holidays] + special_holidays
        holidays.sort()

        return holidays

def nyse_close_early_dates(year=datetime.date.today().year):
    """
    Get dates that the NYSE closes early.
    """
    return [
        datetime.date(year, 6, 3),                 # 1:00pm day before Independence Day
        nth_week_day_of_month(4, 'Wed', 11, year), # 1:00pm day before Thanksgiving Day
        datetime.date(year, 12, 24)                # 1:00pm day before Christmas Day
    ]

def closing_time(date=datetime.date.today()):
    """
    Get closing time of the current date.
    """
    return datetime.time(13, 0) if date in nyse_close_early_dates(date.year) else datetime.time(16, 0)

def opening_time():
    """
    Get opening time of the current date.
    """
    return datetime.time(9, 30)

def is_holiday(date=datetime.date.today()):
    """
    Return boolean if date is a NYSE holiday.
    """
    return date in nyse_holidays(date.year)

def is_open_on(date=datetime.date.today()):
    """
    Return boolean if NYSE is open on this date (not weekend or holiday).
    """
    return not date.weekday() >= 5 or is_holiday(date)

def is_open_at(dt=datetime.datetime.today()):
    """
    Return boolean if the NYSE is open at a specific time (includes normal trading hours, close early days and holidays).
    """
    # If weekend or holiday
    if not is_open_on(dt):
        return False
    else:
        return datetime.time(9, 30) < dt.time() < closing_time(dt.date())

# Market EOD Data Download Functions
# ------------------------------------------------------------------------------------------------------------------------------
def download_all_symbols():
    """
    Download current symbols from NASDAQ server, return as DataFrame.
    """

    # Get NASDAQ symbols
    nasdaq_text = urllib2.urlopen(NASDAQ_URL + NASDAQ_FILE).read()
    # Process NASDAQ symbols
    nasdaq = pd.read_csv(StringIO.StringIO(nasdaq_text), delimiter='|')
    # Drop Unneccesary data (NextShares)
    nasdaq = nasdaq.ix[:, :-1]
    # Drop Unneccesary Row (File Create Date)
    nasdaq = nasdaq.iloc[:-1]
    # Set Exchange and ETFness
    nasdaq['ETF'] = 'N'
    nasdaq['Exchange'] = 'Q'
    # Clean Columns
    nasdaq['NASDAQ Symbol'] = nasdaq['Symbol']

    # Get OTHER (NYSE, BATS) symbols
    other_text = urllib2.urlopen(NASDAQ_URL + OTHERS_FILE).read()
    # Process OTHER symbols
    other = pd.read_csv(StringIO.StringIO(other_text), delimiter='|')
    # Drop Unneccesary Column (NextShares)
    # other = other.ix[:, :-1]
    # Drop Unneccesary Row (File Create Date)
    other = other.iloc[:-1]
    # Clean Columns
    other = other.rename(columns={'ACT Symbol': 'Symbol'})

    # Concatenate NASDAQ and OTHER data frames together
    symbols = pd.concat([nasdaq, other], ignore_index=False)
    symbols = symbols.sort_values(by='Symbol').reset_index(drop=True)
    symbols['Exchange'] = symbols['Exchange'].map(EXCHANGE_ABBR)
    symbols = symbols[COLUMN_ORDER]
    symbols = symbols.set_index('Symbol')

    # Drop unnecesary Columns
    symbols = symbols.drop(['NASDAQ Symbol', 'Test Issue'], 1)

    # Drop Nasdaq test stock symbols (experimentally found)
    symbols = symbols.drop(['ZJZZT', 'ZVZZC', 'ZVZZT', 'ZWZZT', 'ZXZZT', 'ZXYZ.A'])

    return symbols

def download_google_history(symbols, start, end=(datetime.date.today() - datetime.timedelta(days=1))):
    """
    Download daily symbol history from Google servers for specified range.
    Returns DataFrame with Date, Open, Close, Low, High, Volume.
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
        result_string = urllib.urlopen(url).read()
        if result_string.find('Not Found') < 0:
            data = pd.read_csv(StringIO.StringIO(result_string), index_col=0, na_values=['', '-'], parse_dates=True)
            data = data.sort_index()
            if len(data.index) > 0  and data.index[0].year == start.year:
                history = data
            break

    return history

def download_yahoo_history(symbols, start):
    """
    Download daily symbol history from Yahoo servers for specified range.
    Returns DataFrame with Date, Open, Close, Low, High, Volume.
    """

    # Set up empty DataFrame
    history = pd.DataFrame({'Open':[], 'Close':[], 'High':[], 'Low':[], 'Volume':[]})
    history.index.name = 'Date'

    # Check each exchange, bounce out once found
    for exchange in EXCHANGES:
        url_vars = {
            'symbol': exchange + symbols,
            'start' : start.strftime('%Y-%m-%d')
        }
        url = YAHOO_URL.format(**url_vars)
        result_string = urllib.urlopen(url).read()
        if result_string.find('Not Found') < 0:
            data = pd.read_csv(
                StringIO.StringIO(result_string),
                index_col=0,
                na_values=['', '-'],
                parse_dates=True
            ).sort_index()
            if len(data.index) > 0:
                history = data
            break

    return history

def log_message(msg, log_location, log=True, display=True):
    """
    Display and log message.
    """
    # Display on stdout
    if display:
        sys.stdout.write(msg)
        sys.stdout.flush()
    # Log to file
    if log:
        for location in log_location:
            location = location.format(datetime.date.today())
            try:
                with open(location, 'a') as f:
                    f.write(msg)
            except IOError:
                pass
# pylint: disable=too-many-arguments,too-many-branches,too-many-statements
def update_history(
        symbol_manifest_location='./data/symbols.csv',
        history_status_location='./data/history.json',
        log_location='./data/log_{}.txt',
        history_path='./data/history/{}',
        source='google',
        log=True,
        display=True,
        trade_days=True,
        force_day=None
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
        trade_days               : boolean to only attempt downloads on trading days, defaults to True
        force_day                : datetime.date to force a specific end day instead of using yesterday

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

    Google, also anecdotally, releases the EOD data at 6:25am the next day.

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
    # EOD data is not released until the following day so always request yesterday's data, unless a force_day is input
    request_date = force_day if force_day else today - datetime.timedelta(days=1)

    # Only procede with downloads if the market was open or non-trade day override
    if is_open_on(request_date) or not trade_days:

        # Check if data directory exists
        for location in symbol_manifest_location:
            if not os.path.exists(os.path.dirname(location)):
                os.makedirs(os.path.dirname(location))
                log_message(
                    'Data directory does not exists. Creating Directory.\n{}\n'.format(os.path.dirname(location)),
                    log_location,
                    log,
                    display
                )

        # Check if history directory exists
        if not os.path.exists(os.path.dirname(history_path)):
            os.makedirs(os.path.dirname(history_path))
            log_message(
                'History directory does not exists. Creating Directory.\n{}\n'.format(os.path.dirname(history_path)),
                log_location,
                log,
                display
            )

        # Read in History Status
        if os.path.exists(history_status_location[0]):
            with open(history_status_location[0], 'r') as f:
                history_status = json.load(f)
                history_status['request_date'] = datetime.datetime.strptime(history_status['request_date'], '%Y-%m-%d').date()
                try:
                    history_status['last'] = datetime.datetime.strptime(history_status['last'], '%Y-%m-%dT%H:%M:%S.%f')
                except ValueError:
                    history_status['last'] = datetime.datetime.strptime(history_status['last'], '%Y-%m-%dT%H:%M:%S')
                history_status['complete'] = False if history_status['request_date'] < request_date else True

        # New History Generation
        else:
            history_status = {
                'count': 0,                                  # Number of time update_history has been called
                'complete': False,                           # Download complete for the day?
                'last': datetime.datetime.now().isoformat(), # Current last update time
                'day': str(request_date),                    # Date that is being downloaded
                'mode': 'build',                             # Building or updating
                'manifest': False,                           # Manifest available
                'current_symbol': None,                      # Current symbol
                'current_date': None,                        # Current date
                'number_of_symbols': 0,                      # Number of symbols
                'build_downloaded': 0,                       # Number of symbols downloaded during build
                'build_download_attempt': 0,                 # Number of symbols attempted to download during build
                'build_percent_complete': 0.0,               # Percent of symbols completed during build
                'build_percent_attempt': 0.0                 # Percent of symbols attempted during build
            }
            log_message('History Status does not exists. Creating Status.\n', log_location, log, display)

        # If symbol manifest exist enter build or update mode
        if os.path.exists(symbol_manifest_location[0]):
            # Read from disk
            symbol_manifest = pd.read_csv(symbol_manifest_location[0], index_col=0, parse_dates=[4, 5, 6])

            # Build Mode: If past symbol history is not complete, incrementally download history backwardsm
            incomplete_history = symbol_manifest[~symbol_manifest['Current']]
            if len(incomplete_history) > 0:

                history_status['mode'] = 'build'

                # Get first incomplete symbol
                symbol = incomplete_history.index.tolist()[0]

                if source is 'yahoo':
                    # Set end date to request_date
                    end = request_date
                    # Set new start date to a year before end
                    start = earliest_date
                else:
                    # Set end date to request_date if first download or set to last start
                    end = request_date if pd.isnull(symbol_manifest.loc[symbol]['End']) else symbol_manifest.loc[symbol]['Start'].date()
                    # Set new start date to a year before end
                    start = (end + pd.DateOffset(years=-download_offset)).date()
                    # Clip end to earliest_date if start is before it
                    start = earliest_date if start < earliest_date else start
                log_message(
                    '{:%Y-%m-%d %H:%M:%S}: (Build Mode) Downloading {} from {} to {}:'.format(now, symbol, start, end),
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
                        with open(history_path.format(symbol + '.pkl'), 'wb') as f:
                            pickle.dump(data, f, protocol=0)
                    else:
                        # Get current data
                        with open(history_path.format(symbol + '.pkl'), 'r') as f:
                            history = pickle.load(f)
                        # Append data
                        history = history.append(data).sort_index()
                        # Make sure dupicate dates are removed
                        history = history[~history.index.duplicated(keep='first')]
                        # Write to disk
                        with open(history_path.format(symbol + '.pkl'), 'wb') as f:
                            pickle.dump(history, f, protocol=0)
                    # Record start in manifest
                    symbol_manifest.loc[symbol, 'Start'] = data.index[0].date()

                # Record in status
                history_status['current_symbol'] = symbol
                history_status['current_date'] = str(start)

                # Store manifest to disk
                for location in symbol_manifest_location:
                    symbol_manifest.to_csv(location)
                log_message(' {}\n'.format('[ ]' if data.empty else '[x]'), log_location, log, display)

            # Update Mode: If current symbol history is not complete, incrementally download history forwards
            else:
                history_status['mode'] = 'update'

                # Are there any incomplete symbols?
                # Use the last download attempt date, not the actual last data date.
                # This prevents infinite loops on days that don't download successfully.
                incomplete_symbols = symbol_manifest.loc[
                    (symbol_manifest['Attempt'] != request_date) & ~pd.isnull(symbol_manifest['Start'])
                ]
                if len(incomplete_symbols) > 0:
                    # Get first incomplete symbol
                    symbol = incomplete_symbols.index.tolist()[0]
                    # Get new start date as last end date
                    start = symbol_manifest.loc[symbol]['End'].date() + pd.DateOffset(days=1)
                    # Set new end date to a year from start
                    end = (start + pd.DateOffset(years=download_offset)).date()
                    # Clip end to request_date if end is in the future
                    end = request_date if end > request_date else end
                    log_message(
                        '{:%Y-%m-%d %H:%M:%S}: (Update Mode) Downloading {} from {:%Y-%m-%d} to {:%Y-%m-%d}:'.format(now, symbol, start, end),
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
                        with open(history_path.format(symbol + '.pkl'), 'wb') as f:
                            pickle.dump(history, f, protocol=0)

                        # Record last ending in manifest (use last non-NaN price date)
                        symbol_manifest.loc[symbol, 'End'] = history.last_valid_index().date()

                    # Record the request data in the manifest
                    symbol_manifest.loc[symbol, 'Attempt'] = request_date

                    # Record in status
                    history_status['current_symbol'] = symbol
                    history_status['current_date'] = str(start)

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
            symbol_manifest['Attempt'] = None
            symbol_manifest['Current'] = False
            # Store to disk
            for location in symbol_manifest_location:
                symbol_manifest.to_csv(location)
            # Status
            history_status['manifest'] = True

        # Store status to disk at the end of the script
        history_status['last'] = datetime.datetime.now().isoformat()
        history_status['request_date'] = str(request_date)
        history_status['count'] += 1

        # Build Mode Numbers: Update Overall History Counts/Percents
        total_symbols = float(len(symbol_manifest))
        history_status['number_of_symbols'] = total_symbols
        history_status['build_downloaded'] = symbol_manifest['Start'].count()
        history_status['build_download_attempt'] = symbol_manifest['End'].count()
        try:
            history_status['build_percent_complete'] = np.round(100.0 * history_status['build_downloaded'] / total_symbols, 2)
            history_status['build_percent_attempt'] = np.round(100.0 * history_status['build_download_attempt'] / total_symbols, 2)
        except ZeroDivisionError:
            history_status['build_percent_complete'] = np.NaN
            history_status['build_percent_attempt'] = np.NaN

        # Update Mode Numbers: Update Current History Counts/Percents
        history_status['update_downloaded'] = float((symbol_manifest['End'] == request_date).sum())
        history_status['update_download_attempt'] = float((symbol_manifest['Attempt'] == request_date).sum())
        try:
            history_status['update_percent_complete'] = np.round(100.0 * history_status['update_downloaded'] / total_symbols, 2)
            history_status['update_percent_attempt'] = np.round(100.0 * history_status['update_download_attempt'] / total_symbols, 2)
        except ZeroDivisionError:
            history_status['update_percent_complete'] = np.NaN
            history_status['update_percent_attempt'] = np.NaN

        # Write the history status as json
        for location in history_status_location:
            with open(location, 'w') as f:
                json.dump(history_status, f, indent=4, separators=(',', ': '), sort_keys=True)

        # Write the history status as text
        for location in [h.replace('.json', '.txt') for h in history_status_location]:
            with open(location, 'w') as f:
                f.write(tabulate.tabulate(sorted(history_status.items())).replace(' ', '.'))
    else:
        done = True
        log_message(
            'Market was not open on {:%Y-%m-%d}. No EOD data available.'.format(request_date) +
            '\nIf initial (historical) download needed, set trade_days to False.\n',
            log_location,
            log,
            display
        )

    return done
