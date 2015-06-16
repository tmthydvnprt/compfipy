"""
Asset Class

Methods for calculating technical indicators:

Overlays:
---------
[x] Bollinger Bands                      - A chart overlay that shows the upper and lower limits of 'normal' price movements based on the Standard Deviation of prices.
[x] Chandelier Exit                      - A indicator that can be used to set trailing stop-losses for both long and short position.
[x] Ichimoku Clouds                      - A comprehensive indicator that defines support and resistance, identifies trend direction, gauges momentum and provides trading signals.
[x] Keltner Channels                     - A chart overlay that shows upper and lower limits for price movements based on the Average True Range of prices.
[x] Moving Averages                      - Simple and Exponential - Chart overlays that show the 'average' value over time. Both Simple Moving Averages (SMAs) and Exponential Moving Averages (EMAs) are explained.
[x] Moving Average Envelopes             - A chart overlay consisting of a channel formed from simple moving averages.
[x] Parabolic SAR                        - A chart overlay that shows reversal points below prices in an uptrend and above prices in a downtrend.
[x] Pivot Points                         - A chart overlay that shows reversal points below prices in an uptrend and above prices in a downtrend.
[x] Price Channels                       - A chart overlay that shows a channel made from the highest high and lowest low for a given period of time.
[x] Volume by Price                      - A chart overlay with a horizontal histogram showing the amount of activity at various price levels.
[x] Volume-weighted Average Price (VWAP) - An intraday indicator based on total dollar value of all trades for the current day divided by the total trading volume for the current day.
[x] ZigZag                               - A chart overlay that shows filtered price movements that are greater than a given percentage.

Indicators:
-----------
[x] Accumulation Distribution Line       - Combines price and volume to show how money may be flowing into or out of a stock.
[x] Aroon                                - Uses Aroon Up and Aroon Down to determine whether a stock is trending or not.
[x] Aroon Oscillator                     - Measures the difference between Aroon Up and Aroon Down.
[x] Average Directional Index (ADX)      - Shows whether a stock is trending or oscillating.
[x] Average True Range (ATR)             - Measures a stock's volatility.
[x] BandWidth                            - Shows the percentage difference between the upper and lower Bollinger Band.
[x] %B Indicator                         - Shows the relationship between price and standard deviation bands.
[x] Commodity Channel Index (CCI)        - Shows a stock's variation from its 'typical' price.
[x] Coppock Curve                        - An oscillator that uses rate-of-change and a weighted moving average to measure momentum.
[x] Chaikin Money Flow                   - Combines price and volume to show how money may be flowing into or out of a stock. Alternative to Accumulation/Distribution Line.
[x] Chaikin Oscillator                   - Combines price and volume to show how money may be flowing into or out of a stock. Based on Accumulation/Distribution Line.
[x] Price Momentum Oscillator            - An advanced momentum indicator that tracks a stock's rate of change.
[x] Detrended Price Oscillator (DPO)     - A price oscillator that uses a displaced moving average to identify cycles.
[x] Ease of Movement (EMV)               - An indicator that compares volume and price to identify significant moves.
[x] Force Index                          - A simple price-and-volume oscillator.
[x] Know Sure Thing (KST)                - An indicator that measures momentum in a smooth fashion.
[x] Mass Index                           - An indicator that identifies reversals when the price range widens.
[x] MACD                                 - A momentum oscillator based on the difference between two EMAs.
[x] MACD-Histogram                       - A momentum oscillator that shows the difference between MACD and its signal line.
[x] Money Flow Index (MFI)               - A volume-weighted version of RSI that shows shifts is buying and selling pressure.
[x] Negative Volume Index (NVI)          - A cumulative volume-based indicator used to identify trend reversals.
[x] On Balance Volume (OBV)              - Combines price and volume in a very simple way to show how money may be flowing into or out of a stock.
[x] Percentage Price Oscillator (PPO)    - A percentage-based version of the MACD indicator.
[x] Percentage Volume Oscillator         - The PPO indicator applied to volume instead of price.
[x] Rate of Change (ROC)                 - Shows the speed at which a stock's price is changing.
[x] Relative Strength Index (RSI)        - Shows how strongly a stock is moving in its current direction.
[x] StockCharts Tech. Ranks (SCTRs)      - Our relative ranking system based on a stock's technical strength.
[ ] Slope                                - Measures the rise-over-run for a linear regression
[x] Standard Deviation (Volatility)      - A statistical measure of a stock's volatility.
[x] Stochastic Oscillator                - Shows how a stock's price is doing relative to past movements. Fast, Slow and Full Stochastics are explained.
[x] StochRSI                             - Combines Stochastics with the RSI indicator. Helps you see RSI changes more clearly.
[x] TRIX                                 - A triple-smoothed moving average of price movements.
[x] True Strength Index (TSI)            - An indicator that measures trend direction and identifies overbought/oversold levels.
[x] Ulcer Index                          - An indicator designed to measure market risk or volatility.
[x] Ultimate Oscillator                  - Combines long-term, mid-term and short-term moving averages into one number.
[x] Vortex Indicator                     - An indicator designed to identify the start of a new trend and define the current trend.
[x] William %R                           - Uses Stochastics to determine overbought and oversold levels.

Charts :
--------

[x] Gaps                                  - An area of price change in which there were no trades.
[ ] Classify Gaps                         - decide if a gap is [ ] common, [ ] breakaway, [ ] runaway, or [ ] exhaustion
[ ] Double Top Reversal
[ ] Double Bottom Reversal
[ ] Head and Shoulders Top (Reversal)
[ ] Head and Shoulders Bottom (Reversal)
[ ] Falling Wedge (Reversal)
[ ] Rising Wedge (Reversal)
[ ] Rounding Bottom (Reversal)
[ ] Triple Top Reversal
[ ] Triple Bottom Reversal
[ ] Bump and Run Reversal (Reversal)
[ ] Flag, Pennant (Continuation)
[ ] Symmetrical Triangle (Continuation)
[ ] Ascending Triangle (Continuation)
[ ] Descending Triangle (Continuation)
[ ] Rectangle (Continuation)
[ ] Price Channel (Continuation)
[ ] Measured Move - Bullish (Continuation)
[ ] Measured Move - Bearish (Continuation)
[ ] Cup with Handle (Continuation)


[ ] Introduction to Candlesticks          - An overview of candlesticks, including history, formation, and key patterns.
[ ] Candlesticks and Support              - How candlestick chart patterns can mark support levels.
[ ] Candlesticks and Resistance           - How candlestick chart patterns can mark resistance levels.
[ ] Candlestick Bullish Reversal Patterns - Detailed descriptions of bullish reversal candlestick patterns
[ ] Candlestick Bearish Reversal Patterns - Detailed descriptions of common bearish reversal candlestick patterns.
[ ] Candlestick Pattern Dictionary        - A comprehensive list of common candlestick patterns.

[ ] Arms CandleVolume                     - A price chart that merges candlesticks with EquiVolume boxes.
[ ] CandleVolume                          - A price chart that merges candlesticks with volume.
[ ] Elder Impulse System                  - A trading system that color codes the price bars to show signals.
[ ] EquiVolume                            - Price boxes that incorporate volume. How to use and interpret EquiVolume boxes.
[ ] Heikin-Ashi                           - A candlestick method that uses price data from two periods instead one one.
[ ] Kagi Charts                           - How to use and interpret Kagi charts.
[ ] Point and Figure Charts               - How to use and interpret Point and Figure charts.
[ ] Renko Charts                          - How to use and interpret Renko charts.
[ ] Three Line Break Charts               - How to use and interpret Three Line Break charts.

[ ] Andrews' Pitchfork                    - Drawing, adjusting and interpreting this trend channel tool.
[ ] Cycles                                - Steps to finding cycles and using the Cycle Lines Tool.
[o] Fibonacci Retracements                - Defines Fibonacci retracements and shows how to use them to identify reversal zones.
[o] Fibonacci Arcs                        - Shows how Fibonacci Arcs can be used to find reversals.
[ ] Fibonacci Fans                        - Explains what Fibonacci Fans are and how they can be used.
[o] Fibonacci Time Zones                  - Describes Fibonacci Time Zones and how they can be used.
[x] Quandrant Lines                       - Defines Quadrant Lines and shows how they can be used to find future support/resistance zones.
[ ] Raff Regression Channel               - A channel tool based on two equidistant trendlines on either side of a linear regression.
[ ] Speed Resistance Lines                - Shows how Speed Resistance Lines are used on charts.

"""

FIBONACCI_DECIMAL = np.array([0, 0.236, 0.382, 0.5, 0.618, 1])
FIBONACCI_SEQUENCE = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

class Asset( object ):

    def __init__(self, symbol='', data=None):
        self.symbol = symbol
        self.data = data

    def __str__(self):
        return str(self.data)

    @property
    def numberOfDays(self):
        return len(self.Close())

    def plot():
        self.data.plot()

    def Close(self):
        return self.data['Close']
    def c(self): return self.Close()

    def Adj_Close(self):
        return self.data['Adj_Close']
    def ac(self): return self.Adj_Close()

    def Open(self):
        return self.data['Open']
    def o(self): return self.Open()

    def High(self):
        return self.data['High']
    def h(self): return self.High()

    def Low(self):
        return self.data['Low']
    def l(self): return self.Low()

    def Volume(self):
        return self.data['Volume']
    def v(self): return self.Volume()

    def all_indicators(self):
        quadrant_range               = self.quadrant_range()
        bollinger_bands              = self.bollinger_bands()
        chandelier_exit              = self.chandelier_exit()
        ichimoku_clouds              = self.ichimoku_clouds()
        keltner_channels             = self.keltner_channels()
        moving_average_envelopes     = self.moving_average_envelopes()
        parabolic_sar                = self.parabolic_sar()
        pivot_point                  = self.pivot_point()
        fibonacci_pivot_point        = self.fibonacci_pivot_point()
        demark_pivot_point           = self.demark_pivot_point()
        price_channel                = self.price_channel()
        aroon                        = self.aroon()
        price_momentum_oscillator    = self.price_momentum_oscillator()
        know_sure_thing              = self.know_sure_thing()
        macd                         = self.macd()
        negative_volume_index        = self.negative_volume_index()
        percentage_price_oscillator  = self.percentage_price_oscillator()
        percentage_volume_oscillator = self.percentage_volume_oscillator()
        stochastic_oscillator        = self.stochastic_oscillator()
        vortex                       = self.vortex()

        return pd.DataFrame({
            'return'                    : self.returns(),
            'money_flow'                : self.money_flow(),
            'money_flow_volume'         : self.money_flow_volume(),
            'typical_price'             : self.typical_price(),
            'close_to_open_range'       : self.close_to_open_range(),
            'l1_quadrant_range'         : quadrant_range['1'],
            'l2_quadrant_range'         : quadrant_range['2'],
            'l3_quadrant_range'         : quadrant_range['3'],
            'l4_quadrant_range'         : quadrant_range['4'],
            'l5_quadrant_range'         : quadrant_range['5'],
            'true_range'                : self.true_range(),
            'high_low_spread'           : self.high_low_spread(),
            'roc'                       : self.rate_of_change(),
            'upper_bollinger_band'      : bollinger_bands['ub'],
            'center_bollinger_band'     : bollinger_bands['mb'],
            'lower_bollinger_band'      : bollinger_bands['lb'],
            'long_chandelier_exit'      : chandelier_exit['long'],
            'short_chandelier_exit'     : chandelier_exit['short'],
            'conversion_ichimoku_cloud' : ichimoku_clouds['conversion'],
            'base_line_ichimoku_cloud'  : ichimoku_clouds['base'],
            'leadingA_ichimoku_cloud'   : ichimoku_clouds['leadA'],
            'leadingB_ichimoku_cloud'   : ichimoku_clouds['leadB'],
            'lagging_ichimoku_cloud'    : ichimoku_clouds['lag'],
            'upper_keltner_channel'     : keltner_channels['ul'],
            'center_keltner_channel'    : keltner_channels['ml'],
            'lower_keltner_channel'     : keltner_channels['ll'],
            'upper_ma_envelope'         : moving_average_envelopes['uma'],
            'center_ma_envelope'        : moving_average_envelopes['ma'],
            'lower_ma_envelope'         : moving_average_envelopes['lma'],
            'rising_parabolic_sar'      : parabolic_sar['rising'],
            'falling_parabolic_sar'     : parabolic_sar['falling'],
            'p_pivot_point'             : pivot_point['p'],
            's1_pivot_point'            : pivot_point['s1'],
            's2_pivot_point'            : pivot_point['s2'],
            'r1_pivot_point'            : pivot_point['r1'],
            'r2_pivot_point'            : pivot_point['r2'],
            'p_fibonacci_pivot_point'   : fibonacci_pivot_point['p'],
            's1_fibonacci_pivot_point'  : fibonacci_pivot_point['s1'],
            's2_fibonacci_pivot_point'  : fibonacci_pivot_point['s2'],
            's3_fibonacci_pivot_point'  : fibonacci_pivot_point['s3'],
            'r1_fibonacci_pivot_point'  : fibonacci_pivot_point['r1'],
            'r2_fibonacci_pivot_point'  : fibonacci_pivot_point['r2'],
            'r3_fibonacci_pivot_point'  : fibonacci_pivot_point['r3'],
            'p_demark_pivot_point'      : demark_pivot_point['p'],
            's1_demark_pivot_point'     : demark_pivot_point['s1'],
            'r1_demark_pivot_point'     : demark_pivot_point['r1'],
            'high_price_channel'        : price_channel['high'],
            'low_price_channel'         : price_channel['low'],
            'center_price_channel'      : price_channel['center'],
#             'volume_by_price'         : self.volume_by_price(),
            'vwap'                      : self.volume_weighted_average_price(),
            'zigzag'                    : self.zigzag(),
            'adl'                       : self.accumulation_distribution_line(),
            'aroon_up'                  : aroon['up'],
            'aroon_down'                : aroon['down'],
            'aroon_oscillator'          : aroon['oscillator'],
            'adx'                       : self.average_directional_index(),
            'atr'                       : self.average_true_range(),
            'bandwidth'                 : self.bandwidth(),
            '%b'                        : self.percent_b(),
            'cci'                       : self.commodity_channel_index(),
            'coppock_curve'             : self.coppock_curve(),
            'chaikin_money_flow'        : self.chaikin_money_flow(),
            'chaikin_oscillator'        : self.chaikin_oscillator(),
            'pmo'                       : price_momentum_oscillator['pmo'],
            'pmo_signal'                : price_momentum_oscillator['signal'],
            'dpo'                       : self.detrended_price_oscillator(),
            'ease_of_movement'          : self.ease_of_movement(),
            'force_index'               : self.force_index(),
            'kst'                       : know_sure_thing['kst'],
            'kst_signal'                : know_sure_thing['signal'],
            'mass_index'                : self.mass_index(),
            'macd'                      : macd['macd'],
            'macd_signal'               : macd['signal'],
            'macd_hist'                 : macd['hist'],
            'money_flow_index'          : self.money_flow_index(),
            'nvi'                       : negative_volume_index['nvi'],
            'nvi_signal'                : negative_volume_index['signal'],
            'obv'                       : self.on_balance_volume(),
            'ppo'                       : percentage_price_oscillator['ppo'],
            'ppo_signal'                : percentage_price_oscillator['signal'],
            'ppo_hist'                  : percentage_price_oscillator['hist'],
            'pvo'                       : percentage_volume_oscillator['pvo'],
            'pvo_signal'                : percentage_volume_oscillator['signal'],
            'pvo_hist'                  : percentage_volume_oscillator['hist'],
            'rsi'                       : self.relative_strength_index(),
            'sctr'                      : self.stock_charts_tech_ranks(),
            's'                         : self.slope(),
            'volatility'                : self.volatility(),
            '%k_stochastic_oscillator'  : stochastic_oscillator['k'],
            '%d_stochastic_oscillator'  : stochastic_oscillator['d'],
            'stochastic_rsi'            : self.stochastic_rsi(),
            'trix'                      : self.trix(),
            'tsi'                       : self.true_strength_index(),
            'ulcer_index'               : self.ulcer_index(),
            'ultimate_oscillator'       : self.ultimate_oscillator(),
            '+vortex'                   : vortex['+'],
            '-vortex'                   : vortex['-'],
            'william_%r'                : self.william_percent_r(),
        })

    def returns(self, p=None):
        """
        Offset Description
        B   business day frequency
        C   custom business day frequency (experimental)
        D   calendar day frequency
        W   weekly frequency
        M   month end frequency
        BM  business month end frequency
        MS  month start frequency
        BMS business month start frequency
        Q   quarter end frequency
        BQ  business quarter endfrequency
        QS  quarter start frequency
        BQS business quarter start frequency
        A   year end frequency
        BA  business year end frequency
        AS  year start frequency
        BAS business year start frequency
        H   hourly frequency
        T   minutely frequency
        S   secondly frequency
        L   milliseonds
        U   microseconds
        """
        return self.Close().pct_change(freq=p)

    # common price transforms :
    # -------------------------
    def money_flow(self):
        return ((self.Close() - self.Low()) - (self.High() - self.Close())) / (self.High() - self.Low())

    def money_flow_volume(self):
        return self.money_flow() * self.Volume()

    def typical_price(self):
        return (self.High() + self.Low() + self.Close()) / 3

    def close_to_open_range(self):
        return self.Open() - self.Close().shift(1)

    def quadrant_range(self):
        size = self.high_low_spread() / 4
        l1   = self.Low()
        l2   = l1 + size
        l3   = l2 + size
        l4   = l3 + size
        l5   = l4 + size
        return pd.DataFrame({'1': l1, '2': l2, '3': l3, 'l4': l4, 'l5': l5})

    def true_range(self):
        return self.High() - self.Low().shift(1)

    def high_low_spread(self):
        return self.High() - self.Low()

    def rate_of_change(self, n = 20 ):
        return 100.0 * (self.Close() - self.Close().shift(n)) / self.Close().shift(n)
    def roc(self, n = 20): return self.rate_of_change(n)

    # Overlays :
    # ----------
    def bollinger_bands(self, n=20, k=2):
        ma = pd.rolling_mean(self.Close(), n)
        ub = ma + k * pd.rolling_std(self.Close(), n)
        lb = ma - k * pd.rolling_std(self.Close(), n)
        return pd.DataFrame({ 'ub': ub, 'mb': ma, 'lb': lb })

    def chandelier_exit(self, n=22, k=3):
        atr            = self.atr(n)
        nDayHigh       = pd.rolling_max(self.High(), n)
        nDayLow        = pd.rolling_min(self.Low(), n)
        chdlrExirLong  = nDayHigh - k * atr
        chdlrExitShort = nDayLow  - k * atr
        return pd.DataFrame({'long': chdlrExirLong, 'short': chdlrExitShort})

    def ichimoku_clouds(self, n1=9, n2=26, n3=52):
        high       = self.High()
        low        = self.Low()
        conversion = (pd.rolling_max(high, n1) + pd.rolling_min(low, n1)) / 2
        base       = (pd.rolling_max(high, n2) + pd.rolling_min(low, n2)) / 2
        leadingA   = (conversion + base) / 2
        leadingB   = (pd.rolling_max(high, n3) + pd.rolling_min(low, n3)) / 2
        lagging    = self.Close().shift(-n2)
        return pd.DataFrame({ 'conversion' : conversion, 'base': base, 'leadA': leadingA, 'leadB': leadingB, 'lag': lagging })

    def keltner_channels(self, n=20, natr=10):
        atr = self.atr(natr)
        ml  = ema(self.Close(), n)
        ul  = ml + 2 * atr
        ll  = ml - 2 * atr
        return pd.DataFrame({'ul': ul, 'ml': ml, 'll': ll})

    def moving_average_envelopes(self, n=20, k=0.025):
        close = self.Close()
        ma    = sma(close, n)
        uma   = ma + (k * ma)
        lma   = ma - (k * ma
        return pd.DataFrame({'uma': uma, 'ma': ma, 'lma': lma})

    def parabolic_sar(self, step_r=0.02, step_f=0.02, maxAf_r=0.2, maxAf_f=0.2):

        high = self.High()
        low  = self.Low()

        rSar    = pd.TimeSeries(np.zeros(len(high)), index=high.index)
        fSar    = pd.TimeSeries(np.zeros(len(high)), index=high.index)
        ep      = high[0]
        aF      = step_r
        sar     = low[0]
        up      = True

        for i in range(1, len(high)):

            if(up):
                # rising SAR
                ep      = np.max([ep, high[i]])
                aF      = np.min([aF + step_r if (ep == high[i]) else aF, maxAf_r])
                sar     = sar + aF * (ep - sar)
                rSar[i] = sar
            else :
                # falling SAR
                ep      = np.min([ep, low[i]])
                aF      = np.min([aF + step_f if (ep == low[i]) else aF, maxAf_f])
                sar     = sar + aF * (ep - sar)
                fSar[i] = sar

            # trend switch
            if (up and (sar > low[i] or sar > high[i])):
                    up     = False
                    sar    = ep
                    aF     = step_f
            elif (not up and (sar < low[i] or sar < high[i])):
                    up     = True
                    sar    = ep
                    aF     = step_r

        return pd.DataFrame({ 'rising' : rSar, 'falling': fSar})

    def pivot_point(self):
        p  = self.typical_price()
        hl = self.high_low_spread()
        s1 = (2 * p) - self.High()
        s2 = p - hl
        r1 = (2 * p) - self.Low()
        r2 = p + hl
        return pd.DataFrame({ 'p': p, 's1': s1, 's2': s2, 'r1': r1, 'r2': r2 })

    def fibonacci_pivot_point(self):
        p  = self.typical_price()
        hl = self.high_low_spread()
        s1 = p - 0.382 * hl
        s2 = p - 0.618 * hl
        s3 = p -     1 * hl
        r1 = p + 0.382 * hl
        r2 = p + 0.618 * hl
        r3 = p +     1 * hl
        return pd.DataFrame({ 'p': p, 's1': s1, 's2': s2, 's3': s3, 'r1': r1, 'r2': r2, 'r3': r3})

    def demark_pivot_point(self):
        hLc    = self.Close()  < self.Open()
        Hlc    = self.Close()  > self.Open()
        hlC    = self.Close() == self.Open()
        p      = np.zeros(len(self.Close()))
        p[hLc] = self.High()[hLc] + 2 * self.Low()[hLc] + self.Close()[hLc]
        p[Hlc] = 2 * self.High()[Hlc] + self.Low()[Hlc] + self.Close()[Hlc]
        p[hlC] = self.High()[hlC] + self.Low()[hlC] + 2 * self.Close()[hlC]
        s1     = p / 2 - self.High()
        r1     = p / 2 - self.Low()
        p      = p / 4
        return pd.DataFrame({'p': p, 's1': s1, 'r1': r1})

    def price_channel(self, n=20):
        nDayHigh = pd.rolling_max(self.High(), n)
        nDayLow  = pd.rolling_min(self.Low(), n)
        center   = (nDayHigh + nDayLow) / 2
        return pd.DataFrame({'high': nDayHigh, 'low': nDayLow, 'center': center})

    def volume_by_price(self, n=14, blockNum=12):
        close           = self.Close()
        volume          = self.Volume()
        ndayClosingHigh = pd.rolling_max(close, n).bfill()
        ndayClosingLow  = pd.rolling_min(close, n).bfill()
        # compute price blocks: high low range in block number steps
        priceBlocks = pd.DataFrame()
        for low, high, in zip(ndayClosingLow, ndayClosingHigh):
            priceBlocks = priceBlocks.append(pd.DataFrame(np.linspace(low,high,blockNum)).T)
        priceBlocks = priceBlocks.reset_index(drop=True)
        # find correct block for each price, then tally that days volume
        volumeByPrice = pd.DataFrame(np.zeros((len(close),blockNum)))
        for j in range(n-1,len(close)):
            for i,c in enumerate(close[j-(n-1):j+1]):
                block                   = len((c >= priceBlocks.iloc[i,:])[(c >= priceBlocks.iloc[i,:]) == True]) - 1
                block = 0 if block < 0 else block
                volumeByPrice.iloc[j, block] = volume[i] + volumeByPrice.iloc[j, block]
        return volumeByPrice

    def volume_weighted_average_price(self):
        tp =  self.typical_price()
        return (tp * self.Volume()).cumsum() / self.Volume().cumsum()
    def vwap(self): return self.volume_weighted_average_price()

    def zigzag(self, percent=7):
        x         = self.Close()
        zigzag    = pd.TimeSeries( np.zeros(self.numberOfDays), index=x.index)
        lastzig   = x[0]
        zigzag[0] = x[0]
        for i in range(1, self.numberOfDays):
            if (np.abs((lastzig - x[i]) / x[i]) > percent / 100):
                zigzag[i] = x[i]
                lastzig   = x[i]
            else :
                zigzag[i] = None
        return pd.Series.interpolate(zigzag)

    # Indicators :
    # ------------
    def accumulation_distribution_line(self):
        return self.money_flow_volume().cumsum()
    def adl(self): return self.accumulation_distribution_line()

    def aroon(self, n=25):
        high                          = self.High()
        nDayHigh                      = pd.rolling_max(high, n, 0)
        highs                         = high[high == nDayHigh]
        timeSinceLastMax              = (highs.index.values[1:] - highs.index.values[0:-1]).astype('timedelta64[D]').astype(int)
        dayB4High                     = (high==nDayHigh).shift(-1).fillna(False)
        daysSinceHigh                 = pd.TimeSeries(np.nan + np.ones(len(high)), index=high.index)
        daysSinceHigh[dayB4High]      = timeSinceLastMax
        daysSinceHigh[high==nDayHigh] = 0
        daysSinceHigh                 = daysSinceHigh.interpolate('time').astype(int).clip_upper(n)
        low                           = self.Low()
        nDayLow                       = pd.rolling_min(low, n, 0)
        lows                          = low[low == nDayLow]
        timeSinceLastMin              = (lows.index.values[1:] - lows.index.values[0:-1]).astype('timedelta64[D]').astype(int)
        dayB4Low                      = (low == nDayLow).shift(-1).fillna(False)
        daysSinceLow                  = pd.TimeSeries(np.nan + np.ones(len(low)), index=low.index)
        daysSinceLow[dayB4Low]        = timeSinceLastMin
        daysSinceLow[low==nDayLow]    = 0
        daysSinceLow                  = daysSinceLow.interpolate('time').astype(int).clip_upper(n)
        aroonUp                       = 100.0 * ((n - daysSinceHigh) / n)
        aroonDn                       = 100.0 * ((n - daysSinceLow) / n)
        aroonOsc                      = aroonUp - aroonDn
        return pd.DataFrame({'up': aroonUp, 'down': aroonDn, 'oscillator': aroonOsc})

    def average_directional_index(self, n=14):
        tr   = self.true_range()
        pdm  = pd.TimeSeries(np.zeros(len(tr)), index=tr.index)
        ndm  = pd.TimeSeries(np.zeros(len(tr)), index=tr.index)
        pdm[(self.High() - self.High().shift(1)) > (self.Low().shift(1) - self.Low())] = (self.High() - self.High().shift(1))
        ndm[(self.Low().shift(1) - self.Low()) > (self.High() - self.High().shift(1))] = (self.Low().shift(1) - self.Low())
        trN  = ema(tr, n)
        pdmN = ema(pdm, n)
        ndmN = ema(ndm, n)
        pdiN = pdmN / trN
        ndiN = ndmN / trN
        dx   = ((pdiN - ndiN) / (pdiN + ndiN)).abs()
        adx  = ((n-1) * dx.shift(1) + dx) / n
        return adx
    def adx(self, n=14): return self.average_directional_index(n)

    def average_true_range(self, n=14):
        """ this is not a 100% correct - redo """
        tr = self.true_range()
        return ((n-1) * tr.shift(1) + tr) / n
    def atr(self, n=14): return self.average_true_range(n)

    def bandwidth(self, n=20, k=2):
        bb = self.bollinger_bands(n, k)
        return (bb['ub'] - bb['lb']) / bb['mb']

    def percent_b(self, n=20, k=2):
        bb = self.bollinger_bands(n, k)
        return (self.Close().shift(1) - bb['lb']) / (bb['ub'] - bb['lb'])

    def commodity_channel_index(self, n=20):
        tp = self.typical_price()
        return (tp - pd.rolling_mean(tp, n)) / (0.015 * pd.rolling_std(tp, n))
    def cci(self, n=20): return self.commodity_channel_index(n)

    def coppock_curve(self, n1=10, n2=14, n3=11):
        """fix"""
        window = range(n1)
        return pd.rolling_window(self.roc(n2) , window) + self.roc(n3)

    def chaikin_money_flow(self, n=20):
        return pd.rolling_sum((self.money_flow_volume()), n) / pd.rolling_sum(self.Volume(), n)
    def cmf(self, n=20): return self.chaikin_money_flow(n)

    def chaikin_oscillator(self,  n1=3, n2=10):
        return ema(self.adl(), n1) - ema(self.adl(), n2)

    def price_momentum_oscillator(self, n1=20, n2=35, n3=10):
        pmo    = ema(10 * ema((100 * (self.Close() / self.Close().shift(1))) - 100, n2), n1)
        signal = ema(pmo, n3)
        return pd.DataFrame({'pmo': pmo, 'signal': signal})
    def pmo(self, n1=20, n2=35, n3=10): return self.price_momentum_oscillator(n1 , n2, n3)

    def detrended_price_oscillator(self, n=20):
        return self.Close().shift(int(n / 2 + 1) - sma(self.Close(), n)
    def dpo(self, n=20): return self.detrended_price_oscillator(n)

    def ease_of_movement(self, n = 14):
        highLowAvg    = (self.High() + self.Low()) / 2
        distanceMoved =  highLowAvg - highLowAvg.shift(1)
        boxRatio      = (self.Volume() / 100000000) / (self.High() - self.Low())
        emv           = distanceMoved / boxRatio
        return sma(emv, n)

    def force_index(self, n=13):
        forceIndex = self.Close() - self.Close().shift(1) * self.Volume()
        return ema(forceIndex, n)

    def know_sure_thing(self, nSig=9):
        rcma1     = sma(self.roc(10), 10)
        rcma2     = sma(self.roc(15), 10)
        rcma3     = sma(self.roc(20), 10)
        rcma4     = sma(self.roc(30), 15)
        kst       = rcma1 + 2 * rcma2 + 3 * rcma3 + 4 * rcma4
        kstSignal = sma(kst, nSig)
        return pd.DataFrame({'kst': kst, 'signal': kstSignal})
    def kst(self, nSig=9): return self.know_sure_thing( nSig)

    def mass_index(self, n1=9, n2=25):
        ema1        = ema(self.high_low_spread(), n1)
        ema2        = ema(ema1, n1)
        emaRatio    = ema1 / ema2
        return pd.rolling_sum(emaRatio, n2)

    def moving_average_convergence_divergence(self, sn=26, fn=12, nSig=9):
        macd       = ema(self.Close(), fn) - ema(self.Close(), sn)
        macdSignal = ema(macd, nSig)
        macdHist   = macd - macdSignal
        return pd.DataFrame({'macd': macd, 'signal': macdSignal, 'hist': macdHist})
    def macd(self, sn=26, fn=12, nSig=9): return self.moving_average_convergence_divergence(sn, fn, nSig)

    def money_flow_index(self, n=14):
        tp  = self.typical_price()
        rmf = tp * self.Volume()
        pmf = rmf.copy()
        nmf = rmf.copy()
        pmf[pmf < 0] = 0
        nmf[nmf > 0] = 0
        mfr = pd.rolling_sum(pmf, n) / pd.rolling_sum(nmf, n)
        return (100 - (100 / (1 + mfr)))

    def negative_volume_index(self, n=255):
        pctChange                                     = self.returns().cumsum()
        # forward fill when volumes increase with last percent change of a volume decrease day
        pctChange[self.Volume() > self.Volume().shift(1)] = None
        pctChange                                     = pctChange.ffill()
        nvi                                           = 1000 + pctChange
        nviSignal                                     = ema(nvi, n)
        return pd.DataFrame({'nvi': nvi, 'signal': nviSignal})
    def nvi(self, n=255): return self.negative_volume_index(n)

    def on_balance_volume(self):
        pObv = self.Volume().astype(float)
        nObv = (-1 * pObv.copy())
        pObv[self.Close()  < self.Close().shift(1)] = 0
        nObv[self.Close()  > self.Close().shift(1)] = 0
        pObv[self.Close() == self.Close().shift(1)] = None
        nObv[self.Close() == self.Close().shift(1)] = None
        obv  = pObv + nObv
        return obv.ffill().cumsum()
    def obv(self): return self.on_balance_volume()

    def percentage_price_oscillator(self, n1=12, n2=26, n3=9):
        ppo       = 100 * (ema(self.Close(), n1) - ema(self.Close(), n2)) / ema(self.Close(), n2)
        ppoSignal = ema(ppo, n3)
        ppoHist   = ppo - ppoSignal
        return pd.DataFrame({'ppo': ppo, 'signal': ppoSignal, 'hist': ppoHist})
    def ppo(self, n1=12, n2=26, n3=9): return self.percentage_price_oscillator(n1, n2, n3)

    def percentage_volume_oscillator(self, n1=12, n2=26, n3=9):
        pvo       = 100 * (ema(self.Volume(), n1) - ema(self.Volume(), n2)) / ema(self.Volume(), n2)
        pvoSignal = ema(pvo, n3)
        pvoHist   = pvo - pvoSignal
        return pd.DataFrame({'pvo': pvo, 'signal': pvoSignal, 'hist': pvoHist})
    def pvo(self, n1=12, n2=26, n3=9): return self.percentage_volume_oscillator(n1, n2, n3)

    def relative_strength_index(self, n=14):
        change         = self.Close() - self.Close().shift(1)
        gain           = change.copy()
        loss           = change.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss           = -1 * loss
        avgGain        = pd.TimeSeries(np.zeros(len(gain)), index=change.index)
        avgLoss        = pd.TimeSeries(np.zeros(len(loss)), index=change.index)
        avgGain[n]     = gain[0:n].sum() / n
        avgLoss[n]     = loss[0:n].sum() / n
        for i in range(n+1, len(gain)):
            avgGain[i] = (n-1) * (avgGain[i-1] / n) + (gain[i] / n)
            avgLoss[i] = (n-1) * (avgLoss[i-1] / n) + (loss[i] / n)
        rs = avgGain / avgLoss
        return 100 - (100 / (1 + rs))
    def rsi(self, n=14): return self.relative_strength_index(n)

    def stock_charts_tech_ranks(self, n=None, w=None):
        n = n if n else [200, 125, 50, 20, 3, 14 ]
        w = w if w else [0.3, 0.3, 0.15, 0.15, 0.5, 0.5]
        close     = self.Close()
        longMa    = 100 * (1 - close / ema(close, n[0]))
        longRoc   = self.roc(n[1])
        mediumMa  = 100 * (1 - close / ema(close, n[2]))
        mediumRoc = self.roc(n[3])
        ppo       = self.ppo()
        shortPpoM = 100 * ((ppo['hist'] - ppo['hist'].shift(n[4])) / n[4]) / 2
        shortRsi  = self.rsi(n[5])
        return w[0] * longMa + w[1] * longRoc + w[2] * mediumMa + w[3] * mediumRoc + w[4] * shortPpoM + w[5] * shortRsi
    def sctr(self, n=None, w=None): return self.stock_charts_tech_ranks(n, w)

    def slope(self):
        close = self.Close()
        return pd.TimeSeries(np.zeros(len(close)), index=close.index)

    def volatility(self, n=20):
        return pd.rolling_std(self.Close(), n)

    def stochastic_oscillator(self, n=20, n1=3):
        nDayHigh = pd.rolling_max(self.High(), n)
        nDayLow  = pd.rolling_min(self.Low(), n)
        percentK = 100 * (self.Close() - nDayLow) / (nDayHigh - nDayLow)
        percentD = sma(percentK, n1)
        return pd.DataFrame({'k': percentK, 'd': percentD})

    def stochastic_rsi(self, n=20):
        rsi     = self.rsi(n)
        highRsi = pd.rolling_max(rsi, n)
        lowRsi  = pd.rolling_min(rsi, n)
        return (rsi - lowRsi) / (highRsi - lowRsi)

    def trix(self, n=15):
        ema1 = ema(self.Close(), n)
        ema2 = ema(ema1, n)
        ema3 = ema(ema2, n)
        return ema3.pct_change()

    def true_strength_index(self, n1=25, n2=13):
        pc      = self.Close() - self.Close().shift(1)
        ema1    = ema(pc, n1)
        ema2    = ema(ema1, n2)
        absPc   = (self.Close() - self.Close().shift(1)).abs()
        absEma1 = ema(absPc, n1)
        absEma2 = ema(absEma1, n2)
        return 100 * ema2 / absEma2
    def tsi(self, n1=25, n2=13): return self.true_strength_index(n1, n2)

    def ulcer_index(self, n=14):
        percentDrawDown = 100 * (self.Close() - pd.rolling_max(self.Close(), n)) / pd.rolling_max(self.Close(), n)
        return np.sqrt(pd.rolling_sum(percentDrawDown * percentDrawDown, n) / n)

    def ultimate_oscillator(self, n1=7, n2=14, n3=28):
        bp = self.Close() - np.min(self.Low(), self.Close().shift(1))
        tr = pd.DataFrame({'a': self.High(), 'b': self.Close().shift(1)}, index=bp.index).max(1) - pd.DataFrame({'a': self.Low(), 'b': self.Close().shift(1)}, index=bp.index).min(1)
        a1 = pd.rolling_sum(bp, n1) / pd.rolling_sum(tr, n1)
        a2 = pd.rolling_sum(bp, n2) / pd.rolling_sum(tr, n2)
        a3 = pd.rolling_sum(bp, n3) / pd.rolling_sum(tr, n3)
        return 100 * (4 * a1 + 2 * a2 + a3) / (4 + 2 + 1)

    def vortex(self, n=14):
        pvm   = self.High() - self.Low().shift(1)
        nvm   = self.Low() - self.High().shift(1)
        pvm14 = pd.rolling_sum(pvm, n)
        nvm14 = pd.rolling_sum(nvm, n)
        tr    = pd.DataFrame({'a': self.high_low_spread(), 'b': (self.High() - self.Close().shift(1)).abs(), 'c': (self.Low() - self.Close().shift(1)).abs()}, index=pvm.index).max(1)
        tr14  = pd.rolling_sum(tr, n)
        pvi14 = pvm14 / tr14
        nvi14 = nvm14 / tr14
        return pd.DataFrame({'+': pvi14, '-': nvi14})

    def william_percent_r(self, n=14):
        return -100 * (pd.rolling_max(self.High(), n)  - self.Close()) / (pd.rolling_max(self.High(), n) - pd.rolling_min(self.Low(), n))

    # Charting :
    # ------------
    def gaps(self):
        open  = self.Open()
        close = self.Close()
        c2o   = self.close_to_open_range()
        gaps  = pd.TimeSeries(np.zeros(len(close)), index = close.index)
        gaps[open > close.shift()] = c2o
        gaps[open < close.shift()] = c2o
        return gaps

    def fibonacci_retracement(price=0.0, lastprice=0.0):
        return price + FIBONACCI_DECIMAL * (lastprice - price)

    def fibonacci_arc(price=0.0, lastprice=0.0, daysSinceLastPrice=0, nDays=0):
        fibRadius = FIBONACCI_DECIMAL * np.sqrt( np.power(lastprice - price, 2) + np.power(daysSinceLastPrice, 2))
        return price - np.sqrt(np.power(fibRadius, 2) - np.power(nDays, 2) )

    def fibonacci_time(date=dt.date.today()):
        return [dt.date.today() + dt.timedelta(days=d) for d in FIBONACCI_SEQUENCE]

    def speedlines(self, n=20):
        high                            = self.High()
        nDayHigh                        = pd.rolling_max(high, n, 0)
        highs                           = high[high == nDayHigh]
        timeSinceLastMax                = (highs.index.values[1:] - highs.index.values[0:-1]).astype('timedelta64[D]').astype(int)
        dayB4High                       = (high == nDayHigh).shift(-1).fillna(False)
        daysSinceHigh                   = pd.TimeSeries( np.nan + np.ones( len(high) ), index = high.index )
        daysSinceHigh[dayB4High]        = timeSinceLastMax
        daysSinceHigh[high == nDayHigh] = 0
        daysSinceHigh                   = daysSinceHigh.interpolate('time').astype(int).clip_upper(n)

        low                             = self.Low()
        nDayLow                         = pd.rolling_min(low, n, 0)
        lows                            = low[low == nDayLow]
        timeSinceLastMin                = (lows.index.values[1:] - lows.index.values[0:-1]).astype('timedelta64[D]').astype(int)
        dayB4Low                        = (low == nDayLow).shift(-1).fillna(False)
        daysSinceLow                    = pd.TimeSeries(np.nan + np.ones(len(low)), index=low.index)
        daysSinceLow[dayB4Low]          = timeSinceLastMin
        daysSinceLow[low == nDayLow]    = 0
        daysSinceLow                    = daysSinceLow.interpolate('time').astype(int).clip_upper(n)

        trendLength                     = (daysSinceHigh - daysSinceLow)
        trend                           = trendLength
        daysBehind                      = pd.TimeSeries(np.zeros(len(low)), index=low.index)
        daysBehind[trend > 0]           = daysSinceLow
        daysBehind[trend < 0]           = daysSinceHigh
        p                               = pd.TimeSeries(np.nan + np.zeros(len(low)), index=low.index)
        p2_3                            = pd.TimeSeries(np.nan + np.zeros(len(low)), index=low.index)
        p1_3                            = pd.TimeSeries(np.nan + np.zeros(len(low)), index=low.index)
        base                            = pd.TimeSeries(np.nan + np.zeros(len(low)), index=low.index)

        p[trend > 0]                    = nDayLow
        p[trend < 0]                    = nDayHigh
        base[trend > 0]                 = nDayHigh
        base[trend < 0]                 = nDayLow
        p2_3[trend > 0]                 = nDayHigh - ((2.0 / 3.0) * (nDayHigh - nDayLow))
        p2_3[trend < 0]                 = nDayLow  + ((2.0 / 3.0) * (nDayHigh - nDayLow))
        p1_3[trend > 0]                 = nDayHigh - ((1.0 / 3.0) * (nDayHigh - nDayLow))
        p1_3[trend < 0]                 = nDayLow  + ((1.0 / 3.0) * (nDayHigh - nDayLow))
        p                               = p.ffill()
        base                            = base.ffill()
        p2_3                            = p2_3.ffill()
        p1_3                            = p1_3.ffill()
        pSlope                          = pd.TimeSeries(np.nan + np.zeros(len(low)), index=low.index)
        p2_3Slope                       = pd.TimeSeries(np.nan + np.zeros(len(low)), index=low.index)
        p1_3Slope                       = pd.TimeSeries(np.nan + np.zeros(len(low)), index=low.index)
        pSlope[trend > 0]               = ((base - p) / (n + trendLength))
        pSlope[trend < 0]               = ((base - p) / (n - trendLength))
        p2_3Slope[trend > 0]            = ((base - p2_3) / (n + trendLength))
        p2_3Slope[trend < 0]            = ((base - p2_3) / (n - trendLength))
        p1_3Slope[trend > 0]            = ((base - p1_3) / (n + trendLength))
        p1_3Slope[trend < 0]            = ((base - p1_3) / (n - trendLength))
        pSlope                          = pSlope.ffill()
        p2_3Slope                       = p2_3Slope.ffill()
        p1_3Slope                       = p1_3Slope.ffill()
        pNow                            = p    + (pSlope    * daysBehind)
        p2_3Now                         = p2_3 + (p2_3Slope * daysBehind)
        p1_3Now                         = p1_3 + (p1_3Slope * daysBehind)

        return pd.DataFrame({'p': pNow, 'p2/3': p2_3, 'p1/3': p1_3})

    # performance :
    # ---------------
    def sharpe_ratio(self):
        daily_ret = self.returns()
        return np.sqrt(250) * np.mean(daily_ret) / np.std(daily_ret)
