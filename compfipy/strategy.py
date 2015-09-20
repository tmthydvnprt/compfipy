"""
strategy.py

Defines a class that contains a strategy definition and the functions to trade on that strategy.
"""
class Strategy(object):
    """
    Defines a particular trade strategy
    """

    def __init__(self,
        asset,
        trade_function=None,
        init_balance=10000.0,
        init_holding=0.0,
        buy_limit=1.0,
        sell_limit=1.0,
        commission=0.0075,
        commission_min=1.00):
        """
        trade an asset with a specifc trading function
        """
        # assumptions
        self.commission_min = commission_min
        self.commission = commission
        self.buy_limit = buy_limit
        self.sell_limit = sell_limit
        self.init_balance = init_balance
        self.init_holding = init_holding

        # particulars
        self.asset = asset
        self.trade_function = trade_function

        # trade data
        self.holding = pd.Series(np.zeros(asset.c.shape), index=asset.c.index)
        self.cash = pd.Series(np.zeros(asset.c.shape), index=asset.c.index)

        # perform trades
        self.signals = self.generate_signals()
        self.trade()

    def generate_signals(self):
        """
        This functions implements the trading rules defined by the trade function and returns trade signals
        """
        # evalute trading funtion
        signals = self.trade_function(self.asset)

        # scale by buy/sell limits
        signals[signals > 0] = self.buy_limit * signals[signals > 0]
        signals[signals < 0] = self.sell_limit * signals[signals < 0]

        return signals

    def trade(self, init_balance=None, init_holding=0.0):
        """trade the asset based on the buy/sell signals"""

        self.init_balance = init_balance if init_balance else self.init_balance
        self.init_holding = init_holding if init_holding else self.init_holding

        for i, bs in enumerate(self.signals):
            last_holding = self.holding.iloc[i-1] if i > 0 else self.init_holding
            last_cash = self.cash.iloc[i-1] if i > 0 else self.init_balance

            # hold position
            if bs == 0:
                self.holding.iloc[i] = last_holding
                self.cash.iloc[i] = last_cash
            # buy position
            elif bs > 0:
                self.holding.iloc[i] = bs * last_cash / asset.c.iloc[i]
                self.cash.iloc[i] = (1.0 - bs) * last_cash
            # sell position
            elif bs < 0:
                self.holding.iloc[i] = (1.0 - bs) * last_holding
                self.cash.iloc[i] = bs * last_holding * asset.c.iloc[i]

################################################################################################################################

def sma_xover_50_200(asset, cash_buy=1.0, holding_sell=1.0):
    """
    buy when sma(50) crosses above sma(200), sell when sma(50) crosses below sma(200)
    """

    # calculate moving averages
    sma50 = sma(asset.c, 50)
    sma200 = sma(asset.c, 200)

    # determine when crossings occur
    signals = np.sign(np.sign(sma50 - sma200).diff())

    return signals

def buy_and_hold(asset):
    """
    buy on the first day and keep
    """
    signals = pd.Series(np.zeros(asset.c.shape), index=asset.c.index)

    return signals
