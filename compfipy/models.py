"""
models

"""
import math
import pandas as pd
import numpy as np

def convert_to_returns(log_returns=None):
    """convert log returns to normal returns"""
    return np.exp(log_returns)

def convert_to_price(x0=1, log_returns=None):
    """convert log returns to normal returns and calculate value from initial price"""
    returns = convert_to_returns(log_returns)
    prices = pd.concat([pd.Series(x0), returns[:-1]], ignore_index=True)
    return prices.cumprod()

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

    ochlv = pd.DataFrame({'Close':prices})
    ochlv['Open'] = prices + prices * np.random.normal(loc=ochl_mu, scale=ochl_sigma, size=prices.shape)
    ochlv['High'] = prices + prices * np.random.normal(loc=ochl_mu, scale=ochl_sigma, size=prices.shape)
    ochlv['Low'] = prices + prices * np.random.normal(loc=ochl_mu, scale=ochl_sigma, size=prices.shape)
    ochlv['Volume'] = v_mu * np.abs(prices.pct_change(2).shift(-2).ffill()) \
                    + np.random.normal(loc=v_mu, scale=v_sigma, size=prices.shape)

    return ochlv
