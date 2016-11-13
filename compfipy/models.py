"""
models.py

Various Stochastic models of the "market" that provide "fake" asset prices to test on.
"""

import math
import datetime as dt
import pandas as pd
import numpy as np

# Common conversion functions used across all models
# ------------------------------------------------------------------------------------------------------------------------------
def convert_to_returns(log_returns=None):
    """
    Convert log returns to normal returns.
    """
    return np.exp(log_returns)

def convert_to_price(x0=1, log_returns=None):
    """
    Convert log returns to normal returns and calculate value from initial price.
    """
    returns = convert_to_returns(log_returns)
    prices = pd.concat([pd.Series(x0), returns[:-1]], ignore_index=True)
    return prices.cumprod()

# Stochastic Models
# ------------------------------------------------------------------------------------------------------------------------------
def brownian_motion(time=500, delta_t=(1.0 / 252.0), sigma=2):
    """
    Return asset price whose returnes evolve according to brownian motion.
    """
    sqrt_delta_t_sigma = math.sqrt(delta_t) * sigma
    log_returns = pd.Series(np.random.normal(loc=0, scale=sqrt_delta_t_sigma, size=time))
    return log_returns

def geometric_brownian_motion(time=500, delta_t=(1.0 / 252.0), sigma=2, mu=0.5):
    """
    Return asset price whose returnes evolve according to geometric brownian motion.
    """
    wiener_process = brownian_motion(time, delta_t, sigma)
    sigma_pow_mu_delta_t = (mu - 0.5 * math.pow(sigma, 2)) * delta_t
    log_returns = wiener_process + sigma_pow_mu_delta_t
    return log_returns

def jump_diffusion(time=500, delta_t=(1.0 / 252.0), mu=0.0, sigma=0.3, jd_lambda=0.1):
    """
    Return jump diffusion process.
    """
    s_n = 0
    t = 0
    small_lambda = -(1.0 / jd_lambda)
    jump_sizes = pd.Series(np.zeros((time,)))

    while s_n < time:
        s_n += small_lambda * math.log(np.random.uniform(0, 1))
        for j in xrange(0, time):
            if t * delta_t <= s_n * delta_t <= (j+1) * delta_t:
                jump_sizes[j] += np.random.normal(loc=mu, scale=sigma)
                break
        t += 1

    return jump_sizes

def merton_jump_diffusion(time=500, delta_t=(1.0 / 252.0), sigma=2, gbm_mu=0.5, jd_mu=0.0, jd_sigma=0.3, jd_lambda=0.1):
    """
    Return asset price whose returnes evolve according to geometric brownian motion with jump diffusion.
    """
    jd = jump_diffusion(time, delta_t, jd_mu, jd_sigma, jd_lambda)
    gbm = geometric_brownian_motion(time, delta_t, sigma, gbm_mu)
    return gbm + jd

def heston_volatility(time=500, delta_t=(1.0 / 252.0), sigma=2, gbm_mu=0.5, jd_mu=0.0, jd_sigma=0.3, jd_lambda=0.1):
    """
    Return asset price whose returnes evolve according to geometric brownian motion with jump diffusion and non constant
    volatility.
    """
    pass

# Create standard EOD data from price data
# ------------------------------------------------------------------------------------------------------------------------------
def generate_ochlv(prices=None, ochl_mu=0.0, ochl_sigma=0.1, v_mu=100000, v_sigma=math.sqrt(10000)):
    """
    Turn asset price into standard EOD data.
    """
    date_rng = pd.date_range(dt.date.today() - dt.timedelta(days=len(prices)), periods=len(prices), freq='D')
    ochlv = pd.DataFrame({'Close':prices})
    ochlv['Open'] = prices + prices * np.random.normal(loc=ochl_mu, scale=ochl_sigma, size=prices.shape)
    ochlv['High'] = prices + prices * np.random.normal(loc=ochl_mu, scale=ochl_sigma, size=prices.shape)
    ochlv['Low'] = prices + prices * np.random.normal(loc=ochl_mu, scale=ochl_sigma, size=prices.shape)
    ochlv['Volume'] = v_mu * np.abs(prices.pct_change(2).shift(-2).ffill()) \
                    + np.random.normal(loc=v_mu, scale=v_sigma, size=prices.shape)
    ochlv = ochlv.set_index(date_rng)
    return ochlv
