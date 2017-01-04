"""
util.py

General constants and functions that will be used throughout the package.
"""

import numpy as np
import pandas as pd

# Constants
# ------------------------------------------------------------------------------------------------------------------------------

# Time constants
DEFAULT_INITIAL_PRICE = 100.0
DAYS_IN_YEAR = 365.25
DAYS_IN_TRADING_YEAR = 252.0
MONTHS_IN_YEAR = 12.0

# Percent Constants
RISK_FREE_RATE = 0.05

# Trading Signal Constants
FIBONACCI_DECIMAL = np.array([0, 0.236, 0.382, 0.5, 0.618, 1])
FIBONACCI_SEQUENCE = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
RANK_DAYS_IN_TRADING_YEAR = [200, 125, 50, 20, 3, 14]
RANK_PERCENTS = [0.3, 0.3, 0.15, 0.15, 0.5, 0.5]

# Number Formater Functions
# ------------------------------------------------------------------------------------------------------------------------------
def fmtp(x):
    """
    Format as percent.
    """
    return '-' if np.isnan(x) else format(x, '.2%')

def fmtpn(x):
    """
    Format as percent without the sign.
    """
    return '-' if np.isnan(x) else format(100.0 * x, '.2f')

def fmtn(x):
    """
    Format as float.
    """
    return '-' if np.isnan(x) else format(x, '.2f')

def fmttn(x):
    """
    Format as "text" float (Thousand, Million, Billion, etc.).
    """
    abs_x = abs(x)
    if np.isnan(x):
        return '-'
    elif abs_x < 1e3:
        return '{:0.2f}'.format(x)
    elif 1e3 <= abs_x < 1e6:
        return '{:0.2f} k'.format(x / 1e3)
    elif 1e6 <= abs_x < 1e9:
        return '{:0.2f} M'.format(x / 1e6)
    elif 1e9 <= abs_x < 1e12:
        return '{:0.2f} B'.format(x / 1e9)
    elif abs_x >= 1e12:
        return '{:0.2f} T'.format(x / 1e12)

# General Price Helper Functions
# ------------------------------------------------------------------------------------------------------------------------------
def sma(x, n=20):
    """
    Return simple moving average pandas data, x, over interval, n.
    """
    return pd.rolling_mean(x, n)

def ema(x, n=20):
    """
    Return exponential moving average pandas data, x, over interval, n.
    """
    return pd.ewma(x, n)

def calc_returns(x):
    """
    Calculate arithmetic returns of price series.
    """
    return x / x.shift(1) - 1.0

def calc_log_returns(x):
    """
    Calculate log returns of price series.
    """
    return np.log(x / x.shift(1))

def calc_price(x, x0=DEFAULT_INITIAL_PRICE):
    """
    Calculate price from returns series.
    """
    return (x.replace(to_replace=np.nan, value=0) + 1.0).cumprod() * x0

def calc_cagr(x):
    """
    Calculate compound annual growth rate.
    """
    start = x.index[0]
    end = x.index[-1]
    return np.power((x.ix[-1] / x.ix[0]), 1.0 / ((end - start).days / DAYS_IN_YEAR)) - 1.0

def rebase_price(x, x0=DEFAULT_INITIAL_PRICE):
    """
    Convert a series to another initial price.
    """
    return x0 * x / x.ix[0]
