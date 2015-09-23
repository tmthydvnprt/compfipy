"""
strategy.py

Defines a class that contains a strategy definition and the functions to trade on that strategy.
"""
class Strategy(object):
    """
    Defines a particular trade strategy
    """

    def __init__(self, portfolio, commission_min=1.00, commission=0.0075, buy_percent=1.0, sell_percent=1.0):
        """create base Strategy class"""

        self.commission_min = commission_min
        self.commission = commission
        self.portfolio = portfolio
        self.buy_percent = buy_percent
        self.sell_percent = sell_percent
        self.long_open = {symbol:False for symbol in portfolio.assets.keys()}
        self.short_open = {symbol:False for symbol in portfolio.assets.keys()}
        self.display_data = []

        record_headers = ['buy price','buy shares','buy fees','sell price','sell shares','sell fees','gain','profit','loss','return','win/loose']
        self.record = {symbol:pd.DataFrame(columns=record_headers) for symbol in portfolio.assets.keys()}

    def on_date(self, date, i):
        """called for each date of portfolio data, implements trading logic"""
        pass

    def enter_long(self, symbol, date, shares):

        self.long_open[symbol] = True
        self.portfolio.trade(symbol, date, shares, self.commission_min, self.commission)

        record = {
            'buy price' : [self.portfolio.assets[symbol].c[date]],
            'buy shares' : [shares],
            'buy fees' : [max(self.commission_min, abs(self.commission * shares))],
            'sell price' : [None],
            'sell shares' : [None],
            'sell fees' : [None],
            'gain' : [None],
            'profit' : [None],
            'loss' : [None],
            'return' : [None],
            'win/loose' : [None]
        }

        self.record[symbol] = self.record[symbol].append(pd.DataFrame(record), ignore_index=True)

    def exit_long(self, symbol, date, shares):

        self.long_open[symbol] = False
        self.portfolio.trade(symbol, date, -1.0 * shares, self.commission_min, self.commission)

        i = self.record[symbol].index[-1]

        self.record[symbol].loc[i, 'sell price'] = self.portfolio.assets[symbol].c[date]
        self.record[symbol].loc[i, 'sell shares'] = shares
        self.record[symbol].loc[i, 'sell fees'] =  max(self.commission_min, abs(self.commission * shares))

        initial = self.record[symbol].loc[i, 'buy price'] * self.record[symbol].loc[i, 'buy shares']
        final = self.record[symbol].loc[i, 'sell price'] * self.record[symbol].loc[i, 'sell shares']
        fees = self.record[symbol].loc[i, 'sell fees'] + self.record[symbol].loc[i, 'sell fees']
        gain = (final - initial)

        self.record[symbol].loc[i, 'gain'] = gain
        self.record[symbol].loc[i, 'return'] = gain / initial

        self.record[symbol].loc[i, 'profit'] = gain if gain > 0 else 0
        self.record[symbol].loc[i, 'loss'] = (abs(gain) + fees) if gain < 0 else fees

        if gain > 0:
            self.record[symbol].loc[i, 'win/loose'] = 'w'
        elif gain < 0:
            self.record[symbol].loc[i, 'win/loose'] = 'l'
        else:
            self.record[symbol].loc[i, 'win/loose'] = '-'

    def enter_short(self):
        self.short_open[symbol] = True
        pass

    def exit_short(self):
        self.short_open[symbol] = False
        pass

    def run(self):
        """iterates over data"""

        self.before_run()

        for date in self.portfolio.cash.index:
            self.on_date(date)

        self.after_run()

    def before_run(self):
        """called at the start of run"""
        self.display_data = []

    def after_run(self):
        """called at the end of run"""

        print tabulate.tabulate(self.display_data, headers=['trade', 'symbol', 'date', 'price', 'shares', 'value', 'cash'])
        print
        performance = self.calc_performance()

        assets = [' '] + performance.keys()
        perf = [performance.values()[0].keys()] + [s.values() for s in performance.values()]
        perf = [list(x) for x in zip(*perf)]

        print tabulate.tabulate(perf, headers=assets)

    def calc_performance(self):
        """calculate performance"""

        performance = {}
        for symbol in self.portfolio.assets.keys():

            trades = len(self.record[symbol])
            profit = self.record[symbol]['profit'].sum()
            loss = self.record[symbol]['loss'].sum()
            try:
                wins = len(self.record[symbol].groupby('win/loose').groups['w'])
            except:
                wins = 0
            try:
                losses = len(self.record[symbol].groupby('win/loose').groups['l'])
            except:
                losses = 0
            try:
                washes = len(self.record[symbol].groupby('win/loose').groups['-'])
            except:
                washes = 0

            performance[symbol] = {
                'trades': trades,
                'wins': wins,
                'losses': losses,
                'washes': washes,
                'profit': profit,
                'loss': loss,
                'net_profit': profit - loss,
                'profit_factor': profit / loss,
                'percent_profitable': wins / trades,
                'average_trade_net_profit' : (profit - loss) / trades
            }

        return performance



################################################################################################################################

class Sma_Xover(Strategy):
    """
    buy when sma(50) crosses above sma(200), sell when sma(50) crosses below sma(200)
    """

    def on_date(self, date):
        """iterates over dates"""

        for symbol in self.portfolio.assets.keys():

            if len(self.portfolio.assets[symbol].c[:date]) > 200:

                # calculate moving averages
                sma50 = sma(self.portfolio.assets[symbol].c[:date], 50).iloc[-2:]
                sma200 = sma(self.portfolio.assets[symbol].c[:date], 200).iloc[-2:]

                # determine crossover and direction
                xover = np.sign(np.sign(sma50 - sma200).diff()).iloc[-1]

                if xover > 0 and not self.long_open[symbol]:
                    current_price = self.portfolio.assets[symbol].c[date]
                    shares = (self.buy_percent * self.portfolio.cash[date]) / current_price
                    self.enter_long(symbol, date, shares)

                    self.display_data.append(['buy', symbol, str(date.date()), current_price, shares, current_price * shares, self.portfolio.cash[date]])

                elif (xover < 0 and self.long_open[symbol]) or (date == self.portfolio.cash.index[-1] and self.long_open[symbol]):
                    current_price = self.portfolio.assets[symbol].c[date]
                    shares = self.sell_percent * self.portfolio.positions[symbol][date]
                    self.exit_long(symbol, date, shares)

                    self.display_data.append(['sell', symbol, str(date.date()), current_price, shares, current_price * shares, self.portfolio.cash[date]])

def buy_and_hold(asset):
    """
    buy on the first day and keep
    """
    pass
