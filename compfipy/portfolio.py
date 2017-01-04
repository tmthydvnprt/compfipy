# pylint: disable=too-many-public-methods
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
import collections
import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from compfipy.util import RISK_FREE_RATE, MONTHS_IN_YEAR, DAYS_IN_TRADING_YEAR
from compfipy.util import calc_returns, calc_cagr, fmtp, fmtn, fmttn

# General Portfolio Class
# ------------------------------------------------------------------------------------------------------------------------------
class Portfolio(object):
    """
    Define a collection of assets with holdings.
    """

    def __init__(self, assets=None, initial_positions=None, init_cash=10000.0, name='Portfolio'):

        # Portfolio base data
        self.name = name
        self.init_cash = init_cash
        self.assets = assets
        self.stats = {}

        # Create empty tables
        empty_dataframe = pd.DataFrame(
            np.zeros((len(assets[assets.keys()[0]].close), len(assets))),
            columns=[symbol for symbol in assets.keys()],
            index=assets[assets.keys()[0]].close.index
        )
        empty_series = empty_dataframe[empty_dataframe.keys()[0]]

        cash = copy.deepcopy(empty_series)
        trades = copy.deepcopy(empty_dataframe)
        fees = copy.deepcopy(empty_dataframe)
        positions = copy.deepcopy(empty_dataframe)

        # Initial trades/positions/cash if given
        if initial_positions:
            for symbol, shares in initial_positions.items():
                trades[symbol][0] = shares * assets[symbol].close[0]
                positions[symbol][:] = shares
        cash[:] = init_cash

        # Store the data tables
        self.cash = cash
        self.trades = trades
        self.fees = fees
        self.positions = positions

    def summary(self):
        """
        Summarize all the holdings and performance of the portfolio.
        """
        print 'Summary of %s from %s - %s' % (self.stats['name'], self.stats['start'], self.stats['end'])
        print 'Annual risk-free rate considered: %s' %(fmtp(self.stats['yearly_risk_free_return']))
        print '\nSummary:'
        data = [[fmtp(self.stats['total_return']), fmtn(self.stats['daily_sharpe']),
                 fmtp(self.stats['cagr']), fmtp(self.stats['max_drawdown']), fmttn(self.stats['market_cap'])]]
        print tabulate.tabulate(data, headers=['Total Return', 'Sharpe', 'CAGR', 'Max Drawdown', 'Market Cap'])

        print '\nAnnualized Returns:'
        data = [[fmtp(self.stats['mtd']), fmtp(self.stats['three_month']), fmtp(self.stats['six_month']),
                 fmtp(self.stats['ytd']), fmtp(self.stats['one_year']), fmtp(self.stats['three_year']),
                 fmtp(self.stats['five_year']), fmtp(self.stats['ten_year']),
                 fmtp(self.stats['incep'])]]
        print tabulate.tabulate(data, headers=['MTD', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', '10Y', 'Incep.'])

        print '\nPeriodic Returns:'
        data = [
            ['sharpe', fmtn(self.stats['daily_sharpe']), fmtn(self.stats['monthly_sharpe']), fmtn(self.stats['yearly_sharpe'])],
            ['mean', fmtp(self.stats['daily_mean']), fmtp(self.stats['monthly_mean']), fmtp(self.stats['yearly_mean'])],
            ['vol', fmtp(self.stats['daily_vol']), fmtp(self.stats['monthly_vol']), fmtp(self.stats['yearly_vol'])],
            ['skew', fmtn(self.stats['daily_skew']), fmtn(self.stats['monthly_skew']), fmtn(self.stats['yearly_skew'])],
            ['kurt', fmtn(self.stats['daily_kurt']), fmtn(self.stats['monthly_kurt']), fmtn(self.stats['yearly_kurt'])],
            ['best price', fmtp(self.stats['best_day'][0]), fmtp(self.stats['best_month'][0]), fmtp(self.stats['best_year'][0])],
            ['best time', self.stats['best_day'].index[0].strftime('%Y-%m-%d'), self.stats['best_month'].index[0].strftime('%Y-%m-%d'), \
                self.stats['best_year'].index[0].strftime('%Y-%m-%d')],
            ['worst price', fmtp(self.stats['worst_day'][0]), fmtp(self.stats['worst_month'][0]), fmtp(self.stats['worst_year'][0])],
            ['worst time', self.stats['worst_day'].index[0].strftime('%Y-%m-%d'), self.stats['worst_month'].index[0].strftime('%Y-%m-%d'), \
                self.stats['worst_year'].index[0].strftime('%Y-%m-%d')]
            ]
        print tabulate.tabulate(data, headers=['daily', 'monthly', 'yearly'])

        print '\nDrawdowns:'
        data = [
            [fmtp(self.stats['max_drawdown']), fmtp(self.stats['avg_drawdown']),
             fmtn(self.stats['avg_drawdown_days'])]]
        print tabulate.tabulate(data, headers=['max', 'avg', '# days'])

        print '\nMisc:'
        data = [['avg. up month', fmtp(self.stats['avg_up_month'])],
                ['avg. down month', fmtp(self.stats['avg_down_month'])],
                ['up year %', fmtp(self.stats['win_year_perc'])],
                ['12m up %', fmtp(self.stats['twelve_month_win_perc'])]]
        print tabulate.tabulate(data)

        self.plot()

    def plot(self):
        """
        Wrapper for pandas plot().
        """
        name = self.name + ' ' if self.name else self.name

        plt.figure()
        ax = self.close().plot(figsize=(16, 6), title='{}Portfolio Asset Prices'.format(name))

        plt.figure()
        ax = self.returns().plot(figsize=(16, 6), title='{}Portfolio Asset Returns'.format(name))

        plt.figure()
        ax = self.values().plot(figsize=(16, 6), title='{}Portfolio Asset Values'.format(name))

        plt.figure()
        ax = self.total_value().plot(label='Total Asset Value', legend=True, figsize=(16, 6), title='{}Portfolio Balances'.format(name))
        self.total_balance().plot(label='Total Balance', legend=True, ax=ax)
        self.cash.plot(label='Total Cash', legend=True, ax=ax)

        plt.figure()
        ax = self.total_return().plot(figsize=(16, 6), title='{}Portfolio Total Returns'.format(name))

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
    # --------------------------------------------------------------------------------------------------------------------------
    def close(self, date_range=slice(None, None, None)):
        """
        Return closing price for each asset.
        """
        return pd.DataFrame({symbol: asset.close[date_range] for symbol, asset in self.assets.iteritems()})

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
        Return value weight of each asset in portfolio. Alias for self.value_weights().
        """
        return self.value_weights(date_range)

    def value_weights(self, date_range=slice(None, None, None)):
        """
        Return value weight of each asset in portfolio.
        """
        return self.values(date_range).div(self.total_value(date_range), axis=0)

    def share_weights(self, date_range=slice(None, None, None)):
        """
        Return share weight of each asset in portfolio.
        """
        return self.positions[date_range].div(self.positions[date_range].sum(axis=1), axis=0)

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
        return self.values(date_range) - self.cost_bases(date_range)

    def returns(self, date_range=slice(None, None, None)):
        """
        Calculate returns of each assets.
        """
        return 100.0 * self.gains(date_range) / self.cost_bases(date_range)

    def market_caps(self, date_range=slice(None, None, None)):
        """
        Return the Market Cap for each asset.
        TODO: handle real 'non-constant' market cap.
        """
        date_range = date_range
        return pd.Series({symbol: asset.market_cap for symbol, asset in self.assets.iteritems()})

    # Calculate Portfolio totals as sums or weighted sums of individual assets
    # --------------------------------------------------------------------------------------------------------------------------
    def total_value(self, date_range=slice(None, None, None)):
        """
        Calculate portfolio value (sum of asset values).
        """
        return self.values(date_range).sum(axis=1)

    def total_share(self, date_range=slice(None, None, None)):
        """
        Calculate portfolio shares (sum of asset shares).
        """
        return self.positions[date_range].sum(axis=1)

    def total_balance(self, date_range=slice(None, None, None)):
        """
        Calculate portfolio balance (asset values + cash)"""
        return self.total_value(date_range) + self.cash

    def total_cost_basis(self, date_range=slice(None, None, None)):
        """
        Calculate portfolio cost basis (sum of asset cost bases).
        """
        return self.cost_bases(date_range).sum(axis=1)

    def total_gain(self, date_range=slice(None, None, None)):
        """
        Calculate portfolio gain.
        """
        return self.gains(date_range).sum(axis=1)

    def total_return(self, date_range=slice(None, None, None)):
        """
        Calculate portfolio returns.
        """
        return (self.weights(date_range) * self.returns(date_range)).sum(axis=1)

    def price(self, date_range=slice(None, None, None)):
        """
        Calculate the share weighted price of the portfolio.
        """
        return (self.close(date_range) * self.share_weights(date_range)).sum(axis=1)

    def market_cap(self, date_range=slice(None, None, None)):
        """
        Calculate the share weighted market cap of the portfolio.
        """
        return (self.market_caps(date_range) * self.share_weights(date_range)).sum(axis=1)

    def drawdown(self):
        """
        Calucate the drawdown from the highest high.
        """
        # Don't change original data
        draw_down = self.total_value()

        # Fill missing data
        draw_down = draw_down.ffill()

        # Ignore initial NaNs
        draw_down[np.isnan(draw_down)] = -np.Inf

        # Get highest high
        highest_high = draw_down.expanding().max()
        draw_down = (draw_down / highest_high) - 1.0
        return draw_down

    def drawdown_info(self):
        """
        Return table of drawdown data.
        """
        drawdown = self.drawdown()
        is_zero = drawdown == 0

        # Find start and end time
        start = ~is_zero & is_zero.shift(1)
        start = list(start[start].index)
        end = is_zero & (~is_zero).shift(1)
        end = list(end[end].index)

        # Handle no ending
        if len(end) is 0:
            end.append(drawdown.index[-1])

        # Handle startingin drawdown
        if start[0] > end[0]:
            start.insert(0, drawdown.index[0])

        # Handle finishing with drawdown
        if start[-1] > end[-1]:
            end.append(drawdown.index[-1])

        info = pd.DataFrame({
            'start': start,
            'end'  : end,
            'days' : [(e - s).days for s, e in zip(start, end)],
            'drawdown':[drawdown[s:e].min() for s, e in zip(start, end)]
        })

        return info

    # Summary stats
    # --------------------------------------------------------------------------------------------------------------------------
    def calc_stats(self, yearly_risk_free_return=RISK_FREE_RATE):
        """
        Calculate common statistics for this portfolio.
        """
        # pylint: disable=too-many-statements

        monthly_risk_free_return = (np.power(1 + yearly_risk_free_return, 1.0 / MONTHS_IN_YEAR) - 1.0) * MONTHS_IN_YEAR
        daily_risk_free_return = (np.power(1 + yearly_risk_free_return, 1.0 / DAYS_IN_TRADING_YEAR) - 1.0) * DAYS_IN_TRADING_YEAR

        # Sample prices
        daily_value = self.total_value()
        monthly_value = daily_value.resample('M').last()
        yearly_value = daily_value.resample('A').last()

        self.stats = {
            'name' : self.name,
            'start': daily_value.index[0],
            'end': daily_value.index[-1],
            'market_cap' : self.market_cap().iloc[-1],
            'yearly_risk_free_return': yearly_risk_free_return,
            'daily_mean': np.nan,
            'daily_vol': np.nan,
            'daily_sharpe': np.nan,
            'best_day': np.nan,
            'worst_day': np.nan,
            'total_return': np.nan,
            'cagr': np.nan,
            'incep': np.nan,
            'max_drawdown': np.nan,
            'avg_drawdown': np.nan,
            'avg_drawdown_days': np.nan,
            'daily_skew': np.nan,
            'daily_kurt': np.nan,
            'monthly_mean': np.nan,
            'monthly_vol': np.nan,
            'monthly_sharpe': np.nan,
            'best_month': np.nan,
            'worst_month': np.nan,
            'mtd': np.nan,
            'pos_month_perc': np.nan,
            'avg_up_month': np.nan,
            'avg_down_month': np.nan,
            'three_month': np.nan,
            'monthly_skew': np.nan,
            'monthly_kurt': np.nan,
            'six_month': np.nan,
            'ytd': np.nan,
            'one_year': np.nan,
            'yearly_mean': np.nan,
            'yearly_vol': np.nan,
            'yearly_sharpe': np.nan,
            'best_year': np.nan,
            'worst_year': np.nan,
            'three_year': np.nan,
            'win_year_perc': np.nan,
            'twelve_month_win_perc': np.nan,
            'yearly_skew': np.nan,
            'yearly_kurt': np.nan,
            'five_year': np.nan,
            'ten_year':  np.nan,
            'return_table': {}
        }

        if len(daily_value) is 1:
            return

        # Stats with daily prices
        r = calc_returns(daily_value)

        if len(r) < 4:
            return

        self.stats['daily_mean'] = DAYS_IN_TRADING_YEAR * r.mean()
        self.stats['daily_vol'] = np.sqrt(DAYS_IN_TRADING_YEAR) * r.std()
        self.stats['daily_sharpe'] = (self.stats['daily_mean'] - daily_risk_free_return) / self.stats['daily_vol']
        self.stats['best_day'] = r.ix[r.idxmax():r.idxmax()]
        self.stats['worst_day'] = r.ix[r.idxmin():r.idxmin()]
        self.stats['total_return'] = (daily_value[-1] / daily_value[0]) - 1.0
        self.stats['ytd'] = self.stats['total_return']
        self.stats['cagr'] = calc_cagr(daily_value)
        self.stats['incep'] = self.stats['cagr']
        drawdown_info = self.drawdown_info()
        self.stats['max_drawdown'] = drawdown_info['drawdown'].min()
        self.stats['avg_drawdown'] = drawdown_info['drawdown'].mean()
        self.stats['avg_drawdown_days'] = drawdown_info['days'].mean()
        self.stats['daily_skew'] = r.skew()
        self.stats['daily_kurt'] = r.kurt() if len(r[(~np.isnan(r)) & (r != 0)]) > 0 else np.nan

        # Stats with monthly prices
        mr = calc_returns(monthly_value)

        if len(mr) < 2:
            return

        self.stats['monthly_mean'] = MONTHS_IN_YEAR * mr.mean()
        self.stats['monthly_vol'] = np.sqrt(MONTHS_IN_YEAR) * mr.std()
        self.stats['monthly_sharpe'] = (self.stats['monthly_mean'] - monthly_risk_free_return) / self.stats['monthly_vol']
        self.stats['best_month'] = mr.ix[mr.idxmax():mr.idxmax()]
        self.stats['worst_month'] = mr.ix[mr.idxmin():mr.idxmin()]
        self.stats['mtd'] = (daily_value[-1] / monthly_value[-2]) - 1.0 # -2 because monthly[1] = daily[-1]
        self.stats['pos_month_perc'] = len(mr[mr > 0]) / float(len(mr) - 1.0) # -1 to ignore first NaN
        self.stats['avg_up_month'] = mr[mr > 0].mean()
        self.stats['avg_down_month'] = mr[mr <= 0].mean()

        # Table for lookback periods
        self.stats['return_table'] = collections.defaultdict(dict)
        for mi in mr.index:
            self.stats['return_table'][mi.year][mi.month] = mr[mi]
        fidx = mr.index[0]
        try:
            self.stats['return_table'][fidx.year][fidx.month] = (float(monthly_value[0]) / daily_value[0]) - 1
        except ZeroDivisionError:
            self.stats['return_table'][fidx.year][fidx.month] = 0.0
        # Calculate YTD
        for year, months in self.stats['return_table'].items():
            self.stats['return_table'][year][13] = np.prod(np.array(months.values()) + 1) - 1.0

        if len(mr) < 3:
            return

        denominator = daily_value[:daily_value.index[-1] - pd.DateOffset(months=3)]
        self.stats['three_month'] = (daily_value[-1] / denominator[-1]) - 1 if len(denominator) > 0 else np.nan

        if len(mr) < 4:
            return

        self.stats['monthly_skew'] = mr.skew()
        self.stats['monthly_kurt'] = mr.kurt() if len(mr[(~np.isnan(mr)) & (mr != 0)]) > 0 else np.nan

        denominator = daily_value[:daily_value.index[-1] - pd.DateOffset(months=6)]
        self.stats['six_month'] = (daily_value[-1] / denominator[-1]) - 1 if len(denominator) > 0 else np.nan

        # Stats with yearly prices
        yr = calc_returns(yearly_value)

        if len(yr) < 2:
            return

        self.stats['ytd'] = (daily_value[-1] / yearly_value[-2]) - 1.0

        denominator = daily_value[:daily_value.index[-1] - pd.DateOffset(years=1)]
        self.stats['one_year'] = (daily_value[-1] / denominator[-1]) - 1 if len(denominator) > 0 else np.nan

        self.stats['yearly_mean'] = yr.mean()
        self.stats['yearly_vol'] = yr.std()
        self.stats['yearly_sharpe'] = (self.stats['yearly_mean'] - yearly_risk_free_return) / self.stats['yearly_vol']
        self.stats['best_year'] = yr.ix[yr.idxmax():yr.idxmax()]
        self.stats['worst_year'] = yr.ix[yr.idxmin():yr.idxmin()]

        # Annualize stat for over 1 year
        self.stats['three_year'] = calc_cagr(daily_value[daily_value.index[-1] - pd.DateOffset(years=3):])
        self.stats['win_year_perc'] = len(yr[yr > 0]) / float(len(yr) - 1.0)
        self.stats['twelve_month_win_perc'] = (monthly_value.pct_change(11) > 0).sum() / float(len(monthly_value) - (MONTHS_IN_YEAR - 1.0))

        if len(yr) < 4:
            return

        self.stats['yearly_skew'] = yr.skew()
        self.stats['yearly_kurt'] = yr.kurt() if len(yr[(~np.isnan(yr)) & (yr != 0)]) > 0 else np.nan
        self.stats['five_year'] = calc_cagr(daily_value[daily_value.index[-1] - pd.DateOffset(years=5):])
        self.stats['ten_year'] = calc_cagr(daily_value[daily_value.index[-1] - pd.DateOffset(years=10):])

        return
        # pylint: enable=too-many-statements

    def display_stats(self):
        """
        Display talbe of stats.
        """
        stats = [
            ('start', 'Start', 'dt'),
            ('end', 'End', 'dt'),
            ('yearly_risk_free_return', 'Risk-free rate', 'p'),
            (None, None, None),
            ('total_return', 'Total Return', 'p'),
            ('daily_sharpe', 'Daily Sharpe', 'n'),
            ('cagr', 'CAGR', 'p'),
            ('max_drawdown', 'Max Drawdown', 'p'),
            ('market_cap', 'Market Cap', 't'),
            (None, None, None),
            ('mtd', 'MTD', 'p'),
            ('three_month', '3m', 'p'),
            ('six_month', '6m', 'p'),
            ('ytd', 'YTD', 'p'),
            ('one_year', '1Y', 'p'),
            ('three_year', '3Y (ann.)', 'p'),
            ('five_year', '5Y (ann.)', 'p'),
            ('ten_year', '10Y (ann.)', 'p'),
            ('incep', 'Since Incep. (ann.)', 'p'),
            (None, None, None),
            ('daily_sharpe', 'Daily Sharpe', 'n'),
            ('daily_mean', 'Daily Mean (ann.)', 'p'),
            ('daily_vol', 'Daily Vol (ann.)', 'p'),
            ('daily_skew', 'Daily Skew', 'n'),
            ('daily_kurt', 'Daily Kurt', 'n'),
            ('best_day', 'Best Day', 'pp'),
            ('worst_day', 'Worst Day', 'pp'),
            (None, None, None),
            ('monthly_sharpe', 'Monthly Sharpe', 'n'),
            ('monthly_mean', 'Monthly Mean (ann.)', 'p'),
            ('monthly_vol', 'Monthly Vol (ann.)', 'p'),
            ('monthly_skew', 'Monthly Skew', 'n'),
            ('monthly_kurt', 'Monthly Kurt', 'n'),
            ('best_month', 'Best Month', 'pp'),
            ('worst_month', 'Worst Month', 'pp'),
            (None, None, None),
            ('yearly_sharpe', 'Yearly Sharpe', 'n'),
            ('yearly_mean', 'Yearly Mean', 'p'),
            ('yearly_vol', 'Yearly Vol', 'p'),
            ('yearly_skew', 'Yearly Skew', 'n'),
            ('yearly_kurt', 'Yearly Kurt', 'n'),
            ('best_year', 'Best Year', 'pp'),
            ('worst_year', 'Worst Year', 'pp'),
            (None, None, None),
            ('avg_drawdown', 'Avg. Drawdown', 'p'),
            ('avg_drawdown_days', 'Avg. Drawdown Days', 'n'),
            ('avg_up_month', 'Avg. Up Month', 'p'),
            ('avg_down_month', 'Avg. Down Month', 'p'),
            ('win_year_perc', 'Win Year %', 'p'),
            ('twelve_month_win_perc', 'Win 12m %', 'p')
        ]

        data = []
        first_row = ['Stat']
        first_row.extend([self.stats['name']])
        data.append(first_row)

        for k, n, f in stats:
            # Blank row
            if k is None:
                row = [''] * len(data[0])
                data.append(row)
                continue

            row = [n]
            raw = self.stats[k]
            if f is None:
                row.append(raw)
            elif f == 'p':
                row.append(fmtp(raw))
            elif f == 'n':
                row.append(fmtn(raw))
            elif f == 't':
                row.append(fmttn(raw))
            elif f == 'pp':
                row.append(fmtp(raw[0]))
            elif f == 'dt':
                row.append(raw.strftime('%Y-%m-%d'))
            else:
                print 'bad'
            data.append(row)

        print tabulate.tabulate(data, headers='firstrow')
