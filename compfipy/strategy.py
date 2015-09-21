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

    def on_date(self, date):
        """called for each date of portfolio data, implements trading logic"""
        pass

    def enter_long(self, symbol, date, shares):
        self.long_open[symbol] = True
        self.portfolio.trade(symbol, date, shares, self.commission_min, self.commission)

    def exit_long(self, symbol, date, shares):
        self.long_open[symbol] = False
        self.portfolio.trade(symbol, date, -1.0 * shares, self.commission_min, self.commission)

    def enter_short(self):
        self.short_open[symbol] = True
        pass

    def exit_short(self):
        self.short_open[symbol] = False
        pass

    def run(self):
        """iterates over data"""
        for date in self.portfolio.cash.index:
            self.on_date(date)

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
                    print 'buy', symbol, date, current_price, shares, current_price * shares

                elif xover < 0 and self.long_open[symbol]:
                    current_price = self.portfolio.assets[symbol].c[date]
                    shares = self.sell_percent * self.portfolio.positions[symbol][date]
                    self.exit_long(symbol, date, shares)
                    print 'sell', symbol, date, current_price, shares, current_price * shares

def buy_and_hold(asset):
    """
    buy on the first day and keep
    """
    pass
