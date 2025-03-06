import numpy as np
import torch
from scipy.stats import norm
from math import log,sqrt,exp,pi
from scipy.integrate import quad
from scipy.optimize import bisect


#%% Normalization
class Normalization:
    def __init__(self, mean_val=None, std_val=None):
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x):
        # Create new tensor that shares the same storage and requires grad
        normalized = (x - self.mean_val) / self.std_val
        if x.requires_grad:
            normalized.requires_grad_(True)
        return normalized

    def unnormalize(self, x):
        # Create new tensor that shares the same storage and requires grad
        unnormalized = x * self.std_val + self.mean_val
        if x.requires_grad:
            unnormalized.requires_grad_(True)
        return unnormalized


#%% Metrics
def metrics(y, x):
    #x: reference signal
    #y: estimated signal
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    # corrlation
    x_mean = np.mean(x, axis=0, keepdims=True)
    y_mean = np.mean(y, axis=0, keepdims=True)
    x_std = np.std(x, axis=0, keepdims=True)
    y_std = np.std(y, axis=0, keepdims=True)
    corr = np.mean((x-x_mean)*(y-y_mean), axis=0, keepdims=True)/(x_std*y_std)

    # MAP
    map = np.mean(np.abs(x-y), axis=0, keepdims=True)

    # MAPE
    mape = np.mean(np.abs((x - y) / x), axis=0, keepdims=True)


    return torch.tensor(corr), torch.tensor(map), torch.tensor(mape)


def call_option_pricer(spot, strike, maturity, r, vol):
    d1 = (log(spot / strike) + (r + 0.5 * vol * vol) * maturity) / (vol * sqrt(maturity))
    d2 = d1 - vol * sqrt(maturity)

    price = spot * norm.cdf(d1) - strike * exp(-r * maturity) * norm.cdf(d2)
    return price


def put_option_pricer(spot, strike, maturity, r, vol):
    d1 = (log(spot / strike) + (r + 0.5 * vol * vol) * maturity) / (vol * sqrt(maturity))
    d2 = d1 - vol * sqrt(maturity)

    price = -spot * norm.cdf(-d1) + strike * exp(-r * maturity) * norm.cdf(-d2)
    return price


def dN(x):
    ''' Probability density function of standard normal random variable x.'''
    return exp(-0.5*x**2)/sqrt(2*pi)

def N(d):
    ''' Cumulative density function of standard normal random variable x. '''
    return quad(lambda x:dN(x),-20,d,limit=50)[0]

def d1f(St, K, tau, r, sigma):
    ''' Black-Scholes-Merton d1 function.'''
    d1 = (log(St / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt(tau))
    return d1

# calculate greek words

def BSM_delta(row):
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r']   
    sigma = row['impl_volatility']
    d1 = d1f(St, K, tau, r, sigma)
    delta = N(d1)
    return delta

def BSM_gamma(row):
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r'] 
    sigma = row['impl_volatility']  
    d1 = d1f(St, K, tau, r, sigma)
    gamma = dN(d1) / (St * sigma * sqrt(tau))
    return gamma

def BSM_theta(row):
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r'] 
    sigma = row['impl_volatility']  
    cp_flag = row['PC']
    d1 = d1f(St, K, tau, r, sigma)
    d2 = d1 - sigma * sqrt(tau)
    if cp_flag == "C":
        theta = -(St * dN(d1) * sigma / (2 * sqrt(tau)) - r * K * exp(-r * tau) * N(d2))
    if cp_flag == "P":
        theta = -(St * dN(d1) * sigma / (2 * sqrt(tau)) + r * K * exp(-r * tau) * N(d2))
    return theta

def BSM_rho(row):
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r'] 
    sigma = row['impl_volatility']  
    d1 = d1f(St, K, tau, r, sigma)
    d2 = d1 - sigma * sqrt(tau)
    rho = K * tau * exp(-r * tau) * N(d2)
    return rho

def BSM_vega(row):
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r'] 
    sigma = row['impl_volatility']  
    d1 = d1f(St, K, tau, r, sigma)
    vega = St * dN(d1) * sqrt(tau)
    return vega

# calculate implied volatility
def implied_volatility_call(St, K, tau, r, market_price, tol=1e-6): 
    def objective_function_call(sigma, St, K, tau, r, market_price):
        return call_option_pricer(St, K, tau, r, sigma) - market_price
    a = 1e-6   # 波动率的下限
    b = 1.0      # 波动率的上限
    return bisect(lambda sigma: objective_function_call(sigma, St, K, tau, r, market_price), a, b, xtol=tol)

def implied_volatility_put(St, K, tau, r, market_price, tol=1e-6):
    def objective_function_put(sigma, St, K, tau, r, market_price):
        return put_option_pricer(St, K, tau, r, sigma) - market_price
    a = 1e-6  # 波動率下限
    b = 1.0   # 波動率上限
    return bisect(lambda sigma: objective_function_put(sigma, St, K, tau, r, market_price), a, b, xtol=tol)

def calculate_implied_volatility(row):
    try:
        St = row['S']
        K = row['strike_price']
        tau = row['tau']
        r = row['r']   
        market_price = row['option_price']

        if row['PC'] == 'C':
            return implied_volatility_call(St, K, tau, r, market_price)
        elif row['PC'] == 'P':
            return implied_volatility_put(St, K, tau, r, market_price)
        
    except ValueError as e: 
        print(f"Error for row {row}: {e}")
        if row['option_price'] > 0:
            return 1.0  # 使用高极端值
        else:
            return 1e-6  # 使用低极端值
        
    except Exception as e:  
        print(f"Unhandled error for row {row}: {e}")
        return None  

def calculate_theory_price(row):
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r']   
    sigma = row['impl_volatility']
    if row['PC'] == 'C':
        return call_option_pricer(St, K, tau, r, sigma)
    elif row['PC'] == 'P':
        return put_option_pricer(St, K, tau, r, sigma)
