"""
portfolio.py

Define a specifc group of Assets along with functions that pertain to the portfolio.

### Portfolio
- [x] Aggregate specific Assets in Table
- [x] keep track of positions
- [x] keep track of fees
- [x] keep track of cost
- [x] keep track of value
- [x] Enter and Exit position
- [ ] ???Rebalance whole Portfolio???
- [x] Return weights
- [ ] Total Unrealized Performance measures (based on specific time)
- [ ] Total Unrealized Risk measures (based on specific time)
- [ ] Total Unrealized Market Comparisons (based on specific time)
- [ ] Total Realized Performance measures (based on holdings)
- [ ] Total Realized Risk measures (based on holdings)
- [ ] Total Realized Market Comparisons (based on holdings)
- [ ] Summarize Portfolio
- [ ]

"""

import copy
import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# General Portfolio Class
# ------------------------------------------------------------------------------------------------------------------------------
class Portfolio(object):
    """
    Define a collection of assets with holdings.
    """

    def __init__(self, assets=None, initial_positions=None, init_cash=10000.0, name=''):

        self.name = name

        # Create empty tables
        empty_dataframe = pd.DataFrame(
            np.zeros((len(assets[assets.keys()[0]].close), len(assets))),
            columns=[symbol for symbol in assets.keys()],
            index=assets[assets.keys()[0]].close.index
        )
        empty_series = empty_dataframe[empty_dataframe.keys()[0]]

        positions = copy.deepcopy(empty_dataframe)
        trades = copy.deepcopy(empty_dataframe)
        fees = copy.deepcopy(empty_dataframe)
        cash = copy.deepcopy(empty_series)

        # Initial trades/positions/cash if given
        if initial_positions:
            for symbol, shares in initial_positions.items():
                trades[symbol][0] = shares * assets[symbol].close[0]
                positions[symbol][:] = shares
        cash[:] = init_cash

        # Store the data in class
        self.init_cash = init_cash
        self.cash = cash
        self.assets = assets
        self.positions = positions
        self.trades = trades
        self.fees = fees

    def summary(self):
        """
        Summarize all the holdings and performance of the portfolio.
        """

    def plot(self):
        """
        Wrapper for pandas plot().
        """
        name = self.name + ' ' if self.name else self.name

        plt.figure()
        ax = self.close().plot(figsize=(16,6), title='{}Portfolio Asset Prices'.format(name))

        plt.figure()
        ax = self.returns().plot(figsize=(16,6), title='{}Portfolio Asset Returns'.format(name))

        plt.figure()
        ax = self.values().plot(figsize=(16,6), title='{}Portfolio Asset Values'.format(name))

        plt.figure()
        ax = self.total_value().plot(label='Total Asset Value', legend=True, figsize=(16,6), title='{}Portfolio Balances'.format(name))
        self.total_balance().plot(label='Total Balance', legend=True, ax=ax)
        self.cash.plot(label='Total Cash', legend=True, ax=ax)

        plt.figure()
        ax = self.total_return().plot(figsize=(16,6), title='{}Portfolio Total Returns'.format(name))

    def trade(self, symbol='', date=-1, shares=0.0, commission_min=1.0, commission=0.0075):
        """
        Execute a trade and update positions.
        """

        # Determine price of trade
        trade_price = shares * self.assets[symbol].close[date]
        fee = max(commission_min, abs(commission * shares))

        # Update records
        self.cash[date:] = self.cash[date] - trade_price - fee
        self.fees[symbol][date] = fee
        self.trades[symbol][date] = trade_price
        self.positions[symbol][date:] = self.positions[symbol][date] + shares

    # Calculate Asset-wise numbers and statistics
    def close(self, date_range=slice(None, None, None)):
        """
        Return closing price for each asset.
        """
        return pd.DataFrame({symbol: asset.close[date_range] for symbol, asset in self.assets.items()})

    def pct_change(self, date_range=slice(None, None, None)):
        """
        Return closing price returns for each asset.
        """
        return 100.0 * self.close(date_range).pct_change()

    def values(self, date_range=slice(None, None, None)):
        """
        Calculate value of each position (shares * close).
        """
        return self.positions[:][date_range] * self.close(date_range)

    def weights(self, date_range=slice(None, None, None)):
        """
        Return weights of each asset in portfolio.
        """
        return self.values(date_range=date_range).div(self.total_value(date_range=date_range), axis=0)

    def cost_bases(self, date_range=slice(None, None, None)):
        """
        Calculate cost basis of each assets.
        """
        costs = self.trades.cumsum() + self.fees.cumsum()
        return costs[date_range]

    def gains(self, date_range=slice(None, None, None)):
        """
        Calculate gain of each assets.
        """
        return self.values(date_range=date_range) - self.cost_bases(date_range=date_range)

    def returns(self, date_range=slice(None, None, None)):
        """
        Calculate returns of each assets.
        """
        return 100.0 * self.gains(date_range=date_range) / self.cost_bases(date_range=date_range)

    # Calculate Portfolio totals as sums or weighted sums of individual assets
    def total_value(self, date_range=slice(None, None, None)):
        """
        Calculate portfolio value (sum of asset values).
        """
        return self.values(date_range=date_range).sum(axis=1)

    def total_balance(self, date_range=slice(None, None, None)):
        """
        Calculate portfolio balance (asset values + cash)"""
        return self.total_value(date_range=date_range) + self.cash

    def total_cost_basis(self, date_range=slice(None, None, None)):
        """
        Calculate portfolio cost basis (sum of asset cost bases).
        """
        return self.cost_bases(date_range=date_range).sum(axis=1)

    def total_gain(self, date_range=slice(None, None, None)):
        """
        Calculate portfolio gain.
        """
        return self.gains(date_range=date_range).sum(axis=1)

    def total_return(self, date_range=slice(None, None, None)):
        """
        Calculate portfolio returns.
        """
        return (self.weights(date_range=date_range) * self.returns(date_range=date_range)).sum(axis=1)
