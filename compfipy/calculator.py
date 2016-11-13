"""
calculator.py

General functions that would be in a finacial calculator.
"""

import numpy as np

# Typical Financial Calculator Functions
# ------------------------------------------------------------------------------------------------------------------------------
def future_vale(pv=100.0, r=0.07, n=1.0, f=1.0):
    """
    Calculate the future value of pv present value after compounding for n periods at r rate every f frequency.
    """
    return pv * np.exp(1.0 + r / f, n * f)

def present_value(fv=100.0, r=0.07, n=1.0, f=1.0):
    """
    Calculate the present value of fv future value before compounding for n periods at r rate every f frequency.
    """
    return fv / np.exp(1.0 + r / f, n * f)

def rate(fv=100.0, pv=90.0, n=1.0, f=1.0):
    """
    Calculate the rate needed to compound a pv present value into a fv future value compounding over n periods every f
    frequency.
    """
    return f * np.power(fv / pv, 1.0 / (n * f)) - 1.0

def periods(fv=0.0, pv=0.0, r=0.0, f=0.0):
    """
    Calculate the period needed to compound a pv present value into a fv future value compounding at r rate every f frequency.
    """
    return np.log(fv / pv) / (f * np.log(1.0 + r / f))

def effective_return(r=0.07, f=2.0):
    """
    Calculate the annual rate needed to equal an r rate at f frequency.
    """
    return np.power(1.0 + (r / f), f) - 1.0

def annual_return(r=0.07, f=1.0):
    """
    Calculate annual return from semiannual return.
    """
    return np.power(1.0 + r, f) - 1.0

def inflation_adjusted(r=0.07, i=0.03):
    """
    Calculate inflation adjusted returns.
    """
    return (1.0 + r) / (1.0 + i) - 1.0

def gain(xi=100.0, xf=110.0):
    """
    Calculate gain from intial to final value.
    """
    return (xf - xi) / xi

def amortization(p=1000.0, r=0.05, n=10.0, f=1.0):
    """
    Calculate periodic payments needed to pay off p principle at r rate over n periods every f frequency.
    """
    return p * (r / f) / (1.0 - np.power(1.0 + r / f, -f*n))

def cagr(xi=100.0, xf=110.0, n=1.0):
    """
    Calculate compund annual growth rate.
    """
    return np.power(xf / xi, 1.0 / n) - 1.0

def length_of_payment(b=1000.0, p=100.0, apr=0.18):
    """
    Calculate the length of payments of b balance with p payment at apr APR.
    """
    i = apr / 30.0
    return (-1.0 / 30.0) * np.log(1.0 + (b / p)*(1.0 - np.power(1.0 + i, 30.0))) / np.log(1.0 + i)

def annuity(p=100.0, r=0.07, n=10.0, f=1.0):
    """
    Calculate future value based on periodic p investment payment at r rate over n periods every f frequency - check this
    formula.
    """
    return p * ((np.power(1.0 + r / f, n * f) - 1.0) / r / f)
