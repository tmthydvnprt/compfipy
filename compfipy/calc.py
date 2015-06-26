"""
calc.py
"""

def future_vale(pv=100.0, r=0.07, n=1.0, f=1.0):
    """calculate the future value of pv present value after compounding for n periods at r rate every f frequency"""
    return pv * np.exp(1 + r / f, n * f)

def present_value(fv=100.0, r=0.07, n=1.0, f=1.0):
    """calculate the present value of fv future value before compounding for n periods at r rate every f frequency"""
    return future_value / np.exp(1 + r / f, n * f)

def rate(fv=100.0, pv=90.0, n=1.0, f=1.0):
    """calculate the rate needed to compound a pv present value into a fv future value compounding over n periods every f frequency"""
    return f * np.power(fv / pv, 1 / (n * f) ) - 1

def periods(fv=0.0, pv=0.0, r=0.0, f=0.0):
    """calculate the period needed to compound a pv present value into a fv future value compounding at r rate every f frequency"""
    return np.log(fv / pv) / (f * np.log(1 + r / f))

def effective_return(r=0.07, f=2.0):
    """calculate the annual rate needed to equal an r rate at f frequency"""
    return np.power(1 + (r / f), f) - 1

def annual_return(r=0.07, f=1.0):
    """calculate annual return from semiannual return"""
    return np.power(1 + r, f) - 1
