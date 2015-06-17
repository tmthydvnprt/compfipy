"""
models

"""
import math
import pandas as pd
import numpy as np

# helper functions

def next_open_day(date = dt.date.today()):
    """Find the next data the NYSE is open"""
    date = date + dt.timedelta(days=1)
    while not is_open_on(date):
        date = date + dt.timedelta(days=1)
    return date

def move_weekend_holiday(d):
    """If the holiday is part of the weekend, move it"""
    # saturday, make holiday friday before
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    # sunday, make holiday monday after
    elif d.weekday() == 6:
        return d + dt.timedelta(days=1)
    else:
        return d

def nyse_holidays(year=dt.date.today().year):
    """ calulate the holidays of the NYSE for the given year """
    if year < 1817:
        print 'The NYSE was not open in ' + str(year) +'! It was founded in March 8, 1817. Returning empty list []'
        return []
    else:
        typical_holidays = [dt.date(year, 1, 1),                       # New Year's Day
                           nth_week_day_of_month(3, 'Mon', 1, year),   # Martin Luther King, Jr. Day
                           nth_week_day_of_month(3, 'Mon', 2, year),   # Washington's Birthday (President's Day)
                           easter(year) - dt.timedelta(days=2),        # Good Friday
                           nth_week_day_of_month(-1, 'Mon', 5, year),  # Memorial Day
                           dt.date(year, 7, 4),                        # Independence Day
                           nth_week_day_of_month(1, 'Mon', 9, year),   # Labor Day
                           nth_week_day_of_month(4, 'Thu', 11, year),  # Thanksgiving Day
                           dt.date(year, 12, 25 )]                     # Christmas Day

        historical_holidays = [dt.date(2012,10,29),  # hurricane sandy
                               dt.date(2012,10,30),  # hurricane sandy
                              ]
        # grab historical holidays for the year
        special_holidays = [v for v in historical_holidays if v.year == year]

        # alter weekend holidays and add special holidays
        holidays = map(move_weekend_holiday, typical_holidays) + special_holidays
        holidays.sort()

        return holidays

def nyse_close_early_dates(year=dt.date.today().year):
    """nyse close early dates"""
    return [dt.date(year, 6, 3),                       # 1:00pm day before Independence Day
            nth_week_day_of_month(4, 'Wed', 11, year), # 1:00pm day before Thanksgiving Day
            dt.date(year, 12, 24)]                     # 1:00pm day before Christmas Day

def closing_time(date=dt.date.today()):
    """closing time"""
    return dt.time(13, 0) if date in nyse_close_early_dates(date.year) else dt.time(16, 0)

def is_holiday(date=dt.date.today()):
    """is holiday"""
    return date in nyse_holidays(date.year)

def is_open_on(date=dt.date.today()):
    """is open on"""
    #if not weekend or holiday
    return not date.weekday() >= 5 or is_holiday(date)

def is_open_at(datetime=dt.datetime.today()):
    """is open at"""
    #if weekend or holiday
    if datetime.weekday() >= 5 or is_holiday(datetime.date()):
        return False
    else:
        return dt.time(9, 30) < datetime.time() < closing_time(datetime.date())

# common conversion functions used across all models
def convert_to_returns(log_returns=None):
    """convert log returns to normal returns"""
    return np.exp(log_returns)

def convert_to_price(x0=1, log_returns=None):
    """convert log returns to normal returns and calculate value from initial price"""
    returns = convert_to_returns(log_returns)
    prices = pd.concat([pd.Series(x0), returns[:-1]], ignore_index=True)
    return prices.cumprod()

# models
def brownian_motion(time=500, dt=(1.0 / 252.0), sigma=2):
    """ return asset price whose returnes evolve according to brownian motion"""

    sqrt_dt_sigma = math.sqrt(dt) * sigma
    log_returns = pd.Series(np.random.normal(loc=0, scale=sqrt_dt_sigma, size=time))
    return log_returns

def geometric_brownian_motion(time=500, dt=(1.0 / 252.0), sigma=2, mu=0.5):
    """ return asset price whose returnes evolve according to geometric brownian motion"""
    wiener_process = brownian_motion(time, dt, sigma)
    sigma_pow_mu_dt = (mu - 0.5 * math.pow(sigma, 2)) * dt
    log_returns = wiener_process + sigma_pow_mu_dt
    return log_returns

def jump_diffusion(time=500, dt=(1.0 / 252.0), mu=0.0, sigma=0.3, jd_lambda=0.1):
    """ return jump diffusion process"""
    s_n = 0
    t = 0
    small_lambda = -(1.0 / jd_lambda)
    jump_sizes = pd.Series(np.zeros((time,)))

    while s_n < time:
        s_n += small_lambda * math.log(np.random.uniform(0, 1))
        for j in xrange(0, time):
            if t * dt <= s_n * dt <= (j+1) * dt:
                jump_sizes[j] += np.random.normal(loc=mu, scale=sigma)
                break
        t += 1

    return jump_sizes

def merton_jump_diffusion(time=500, dt=(1.0 / 252.0), sigma=2, gbm_mu=0.5, jd_mu=0.0, jd_sigma=0.3, jd_lambda=0.1):
    """ return asset price whose returnes evolve according to geometric brownian motion with jump diffusion"""
    jd = jump_diffusion(time, dt, jd_mu, jd_sigma, jd_lambda)
    gbm = geometric_brownian_motion(time, dt, sigma, gbm_mu)
    return gbm + jd

# def heston_volatility(time=500, dt=(1.0 / 252.0), sigma=2, gbm_mu=0.5, jd_mu=0.0, jd_sigma=0.3, jd_lambda=0.1):
#     """ return asset price whose returnes evolve according to
#     geometric brownian motion with jump diffusion and non constant volatility"""
#     pass

def generate_ochlv(prices=None, ochl_mu=0.0, ochl_sigma=0.1, v_mu=100000, v_sigma=math.sqrt(10000)):
    """turn asset price into standard data"""
    date_rng = pd.date_range(dt.date.today() - dt.timedelta(days=len(prices)), periods=len(prices), freq='D')
    ochlv = pd.DataFrame({'Close':prices})
    ochlv['Open'] = prices + prices * np.random.normal(loc=ochl_mu, scale=ochl_sigma, size=prices.shape)
    ochlv['High'] = prices + prices * np.random.normal(loc=ochl_mu, scale=ochl_sigma, size=prices.shape)
    ochlv['Low'] = prices + prices * np.random.normal(loc=ochl_mu, scale=ochl_sigma, size=prices.shape)
    ochlv['Volume'] = v_mu * np.abs(prices.pct_change(2).shift(-2).ffill()) \
                    + np.random.normal(loc=v_mu, scale=v_sigma, size=prices.shape)
    ochlv = ochlv.set_index(date_rng)
    return ochlv
