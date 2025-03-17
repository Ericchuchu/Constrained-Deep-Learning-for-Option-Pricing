import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import brentq


#%% Normalization
class Normalization:
    def __init__(self, mean_val=None, std_val=None):
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x):
        # Create new tensor that shares the same storage and requires grad
        device = x.device
        self.std_val = self.std_val.to(device)
        self.mean_val = self.mean_val.to(device)
        normalized = (x - self.mean_val) / self.std_val
        if x.requires_grad:
            normalized.requires_grad_(True)
        return normalized

    def unnormalize(self, x):
        # Create new tensor that shares the same storage and requires grad
        device = x.device
        self.std_val = self.std_val.to(device)
        self.mean_val = self.mean_val.to(device)
        unnormalized = x * self.std_val + self.mean_val
        if x.requires_grad:
            unnormalized.requires_grad_(True)
        return unnormalized


#%% Metrics
def metrics(y, x):
    #x: reference signal
    #y: estimated signal
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(y):
        y = y.cpu().numpy()

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
    """計算看漲期權的理論價格 (Black-Scholes 模型)"""
    # 處理邊界條件
    if maturity <= 0 or vol <= 0:
        return max(0, spot - strike * np.exp(-r * maturity))
    
    d1 = (np.log(spot / strike) + (r + 0.5 * vol**2) * maturity) / (vol * np.sqrt(maturity))
    d2 = d1 - vol * np.sqrt(maturity)
    
    # 使用SciPy的norm.cdf提高效率和穩定性
    price = spot * norm.cdf(d1) - strike * np.exp(-r * maturity) * norm.cdf(d2)
    return price

def put_option_pricer(spot, strike, maturity, r, vol):
    """計算看跌期權的理論價格 (Black-Scholes 模型)"""
    # 處理邊界條件
    if maturity <= 0 or vol <= 0:
        return max(0, strike * np.exp(-r * maturity) - spot)
    
    # 使用put-call parity計算可以提高數值穩定性
    call_price = call_option_pricer(spot, strike, maturity, r, vol)
    price = call_price - spot + strike * np.exp(-r * maturity)
    return price

def d1f(St, K, tau, r, sigma):
    """計算Black-Scholes-Merton d1函數，處理邊界條件"""
    if tau <= 0 or sigma <= 0:
        # 處理極限情況
        if St > K:
            return np.inf  # 深度價內
        elif St < K:
            return -np.inf  # 深度價外
        else:
            return 0  # 平價
    
    d1 = (np.log(St / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return d1

# calculate greek words
def BSM_delta(row):
    """計算Delta希臘字母"""
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r']   
    sigma = row['impl_volatility']
    cp_flag = row['PC']
    
    if tau <= 0:
        # 到期時的Delta處理
        if cp_flag == 'C':
            return 1.0 if St > K else 0.0
        else:
            return -1.0 if St < K else 0.0
    
    d1 = d1f(St, K, tau, r, sigma)
    
    if cp_flag == 'C':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0  # put delta = call delta - 1
    
def BSM_gamma(row):
    """計算Gamma希臘字母"""
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r'] 
    sigma = row['impl_volatility']
    
    # 處理邊界條件
    if tau <= 0 or sigma <= 0:
        return 0.0
    
    d1 = d1f(St, K, tau, r, sigma)
    # 使用SciPy的norm.pdf代替自定義的dN函數
    gamma = norm.pdf(d1) / (St * sigma * np.sqrt(tau))
    return gamma

def BSM_theta(row):
    """計算Theta希臘字母"""
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r'] 
    sigma = row['impl_volatility']  
    cp_flag = row['PC']
    
    # 處理邊界條件
    if tau <= 0:
        return 0.0
    
    d1 = d1f(St, K, tau, r, sigma)
    d2 = d1 - sigma * np.sqrt(tau)
    
    # 使用SciPy的函數計算
    common_term = -(St * norm.pdf(d1) * sigma) / (2 * np.sqrt(tau))
    
    if cp_flag == 'C':
        theta = common_term - r * K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        theta = common_term + r * K * np.exp(-r * tau) * norm.cdf(-d2)
    
    return theta

def BSM_rho(row):
    """計算Rho希臘字母"""
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r'] 
    sigma = row['impl_volatility']
    cp_flag = row['PC']
    
    # 處理邊界條件
    if tau <= 0:
        return 0.0
    
    d1 = d1f(St, K, tau, r, sigma)
    d2 = d1 - sigma * np.sqrt(tau)
    
    if cp_flag == 'C':
        rho = K * tau * np.exp(-r * tau) * norm.cdf(d2)
    else:
        rho = -K * tau * np.exp(-r * tau) * norm.cdf(-d2)
        
    return rho

def BSM_vega(row):
    """計算Vega希臘字母"""
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r'] 
    sigma = row['impl_volatility']
    
    # 處理邊界條件
    if tau <= 0:
        return 0.0
    
    d1 = d1f(St, K, tau, r, sigma)
    vega = St * norm.pdf(d1) * np.sqrt(tau)
    return vega

def implied_volatility(St, K, tau, r, market_price, cp_flag, tol=1e-8, max_iterations=100):
    """使用改進的Brent方法計算隱含波動率，增加數值穩定性"""
    # 處理邊界條件
    if tau <= 0:
        return 0.0
    
    # 設定初始搜索範圍
    sigma_min = 0.001
    sigma_max = 5.0  # 更寬的搜索範圍
    
    # 檢查價格是否在理論界限內
    intrinsic = max(0, St - K * np.exp(-r * tau)) if cp_flag == 'C' else max(0, K * np.exp(-r * tau) - St)
    
    if market_price < intrinsic:
        return sigma_min  # 價格低於內在價值
    
    if cp_flag == 'C' and market_price >= St:
        return sigma_max  # 看漲期權價格接近或超過股價
    
    if cp_flag == 'P' and market_price >= K:
        return sigma_max  # 看跌期權價格接近或超過執行價
    
    # 定義目標函數
    def objective(sigma):
        if cp_flag == 'C':
            return call_option_pricer(St, K, tau, r, sigma) - market_price
        else:
            return put_option_pricer(St, K, tau, r, sigma) - market_price
    
    try:
        # 使用Brent方法求解，更精確、更穩定
        return brentq(objective, sigma_min, sigma_max, xtol=tol, maxiter=max_iterations)
    except ValueError:
        # 擴大搜索範圍重試
        try:
            return brentq(objective, 0.0001, 10.0, xtol=tol, maxiter=max_iterations)
        except:
            # 如果仍無法找到解，使用備用方法
            if market_price > intrinsic * 3:
                return sigma_max  # 價格異常高
            else:
                return sigma_min  # 價格接近內在價值

def calculate_implied_volatility(row):
    """計算期權的隱含波動率，增強錯誤處理"""
    try:
        St = row['S']
        K = row['strike_price']
        tau = row['tau']
        r = row['r']   
        market_price = row['option_price']
        cp_flag = row['PC']

        # 數值檢查
        if tau < 0 or St <= 0 or K <= 0:
            return None
        
        # 檢查市場價格是否為零或負數
        if market_price <= 0:
            return 0.001  # 極小值
        
        return implied_volatility(St, K, tau, r, market_price, cp_flag)
        
    except Exception as e:  
        print(f"Error calculating IV for row {row.name if hasattr(row, 'name') else ''}: {e}")
        # 根據期權價格返回合理默認值
        if tau < 0.01:  # 接近到期
            return 0.001
        elif 'moneyness' in row and row['moneyness'] > 1.1 or row['moneyness'] < 0.9:  # 深度價內或價外
            return 0.5
        else:
            return 0.2  # 接近平價

def calculate_theory_price(row):
    """計算期權的理論價格"""
    St = row['S']
    K = row['strike_price']
    tau = row['tau']
    r = row['r']   
    sigma = row['impl_volatility']
    cp_flag = row['PC']
    
    # 數值檢查
    if tau < 0 or sigma <= 0 or St <= 0 or K <= 0:
        # 返回內在價值
        if cp_flag == 'C':
            return max(0, St - K * np.exp(-r * max(0, tau)))
        else:
            return max(0, K * np.exp(-r * max(0, tau)) - St)
    
    if cp_flag == 'C':
        return call_option_pricer(St, K, tau, r, sigma)
    else:
        return put_option_pricer(St, K, tau, r, sigma)
    
def put_to_call_parity(row):
    """使用put-call平價關係將put轉換為call"""
    # 從row中提取所需數據
    put_price = row['option_price']
    stock_price = row['S']
    strike_price = row['strike_price']
    risk_free_rate = row['r']
    time_to_expiry = row['tau']
    
    # 數值檢查
    if time_to_expiry < 0:
        time_to_expiry = 0
    
    # 計算折現因子
    discount_factor = np.exp(-risk_free_rate * time_to_expiry)
    
    # 應用put-call平價公式
    call_price = put_price + stock_price - strike_price * discount_factor
    
    return max(0, call_price)  # 確保不返回負值