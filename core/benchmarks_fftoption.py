import QuantLib as ql
from scipy import stats
from fftoptionlib.fourier_pricer import carr_madan_fft_call_pricer
from fftoptionlib.helper import spline_fitting
from fftoptionlib.process_class import (
    BlackScholes,
    KouJump,
    VarianceGamma,
)
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib
# matplotlib.use('Agg') # TODO: uncomment
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

def compute_func(x):
    return x.compute() if 'dask' in str(type(x)) else x

def filter(invm, spec, threshold=1):
    # parse in the money, out the money, at the money
    if 'dask' in str(type(invm)):
        mask = invm < -999999
    else:
        mask = np.zeros_like(invm, dtype=bool)
    if 'i' in spec:
        mask = mask | (invm < threshold)
    if 'o' in spec:
        mask = mask | (invm > threshold)
    return mask

def BS_BoonHong(S0, K, tau, r, sigma, is_call):
    """
    S0 ：spot price
    K : strike price
    tau : time (year)
    r : free-risk rate
    sigma : vol
    is_call : tau/F
    """
    if is_call:
        d1 = (np.log(S0 / K) + (r + sigma * sigma / 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        return S0 * stats.norm.cdf(d1) - K * np.exp(-r * tau) * stats.norm.cdf(d2)

    else:
        d1 = (np.log(S0 / K) + (r + sigma * sigma / 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        return - S0 * stats.norm.cdf(-d1) + K * np.exp(-r * tau) * stats.norm.cdf(-d2)

def compute_call_price_by_parametric_model(N, eta, alpha, S0, r, d, tau, strike_price_list, mode, *params):
    global sim_k_arr, sim_c_arr
    if mode=="bs":
        sim_k_arr, sim_c_arr = carr_madan_fft_call_pricer(N, eta, alpha, r, tau, S0, d,
                                                          BlackScholes(*params).set_type('chf'))
    elif mode=="vg":
        sim_k_arr, sim_c_arr = carr_madan_fft_call_pricer(N, eta, alpha, r, tau, S0, d,
                                                          VarianceGamma(*params).set_type('chf'))
    elif mode== "kj":
        sim_k_arr, sim_c_arr = carr_madan_fft_call_pricer(N, eta, alpha, r, tau, S0, d,
                                                          KouJump(*params).set_type('chf'))
    ffn_pricer = spline_fitting(sim_k_arr, sim_c_arr, 3)
    predicted_price_list = ffn_pricer(strike_price_list)
    return predicted_price_list

def general_loss(N, eta, alpha, dataset, mode, *params):
    square_error = []
    absolute_percentage_error = []

    for _, df in dataset.groupby(['r', 'tau']):
        S0 = df['S'].iloc[0]
        r = df['r'].iloc[0]
        d = df['d'].iloc[0]
        tau = df['tau'].iloc[0]
        strike_price_list = (df['strike_price']).values
        gold_price_list = df['option_price'].values
        predicted_price_list = compute_call_price_by_parametric_model(N, eta, alpha, S0, r, d, tau, strike_price_list, mode, *params)
        square_error = square_error + list(np.power(gold_price_list - predicted_price_list, 2))
        absolute_percentage_error = absolute_percentage_error + list(np.abs(gold_price_list - predicted_price_list)/gold_price_list)
    return np.mean(square_error), np.mean(absolute_percentage_error)

def sse(N, eta, alpha, dataset, mode):
    def cali_loss_func(*params):
        # only accept the parameters of pricing models
        MSE, MAPE = general_loss(N, eta, alpha, dataset, mode, *params)
        return MSE+100*MAPE
    #func = lambda vol: np.power(gold_price_list - BS_BoonHong(S0, K, tau, r, vol, is_call),2).mean()
    return cali_loss_func

def optimization(N, eta, alpha, df, mode):
    # https://www.cnblogs.com/xuruilong100/p/9919091.html
    constraint = ql.NoConstraint()
    MINVALUE, MAXVALUE = -10000, 10000
    if mode=="bs":
        constraint = ql.PositiveConstraint()
        init = ql.Array(1, 0.15) #volatility (sigma) > 0
    elif mode=="vg":
        constraint = ql.NonhomogeneousBoundaryConstraint([MINVALUE, 0, 0], [MAXVALUE, MAXVALUE, MAXVALUE])
        init = ql.Array(3)
        init[0] = -0.14 #the skewness of distribution of log underlying price (theta), no constraint
        init[1] = 0.2 #the variance rate of the gamma time change (v) > 0
        init[2] = 0.12 #volatility (sigma) > 0
    elif mode=="kj":
        # the shape of characteristic function of Kou Jump refer to :
        # 1. page one of http://faculty.baruch.cuny.edu/jgatheral/jumpdiffusionmodels.pdf
        # 2. https://demonstrations.wolfram.com/OptionPricesInTheKouJumpDiffusionModel/
        # 3. http://www.columbia.edu/~sk75/MagSci02.pdf
        # 4. http://www.optioncity.net/pubs/explevy.pdf
        init = ql.Array(5)
        constraint = ql.NonhomogeneousBoundaryConstraint([0, 0, 1, 0, 0], [MAXVALUE, MAXVALUE, MAXVALUE, MAXVALUE, 1])
        init[0] = 0.15 #volatility (sigma) > 0
        init[1] = 23.34 # jump intensity parameter of Poisson process (jump_rate, lambda) > 0
        init[2] = 59.37 #expected positive jump sizes (exp_pos, or alpha_plus) > 1
        init[3] = 59.45 #-59.45 #expected negative jump sizes (exp_neg, or -alpha_minus) < 0 # try > 0
        init[4]= 0.1 #-200.08 #0 < relative probability of a positive jump (p, prob_pos) < 1

    maxIterations = 10000
    minStatIterations = 9999
    rootEpsilon = 1e-10 # 1e-16
    functionEpsilon = 1e-10 # 1e-16
    gradientNormEpsilon = 1e-10 # 1e-16

    myEndCrit = ql.EndCriteria(maxIterations , minStatIterations , rootEpsilon , functionEpsilon ,
    gradientNormEpsilon)

    er = sse(N, eta, alpha, df, mode)

    out = ql.Optimizer().solve(function=er,c=constraint,e=myEndCrit,m=ql.Simplex(1.0),iv=init)
    calibrated_params = np.array(out)
    return calibrated_params

def read_dataset(dataset_path = "../Dataset/prs_dataset.csv"):
    if os.path.isdir(dataset_path):
        prs_dataset = dd.read_csv(dataset_path + "/*", parse_dates=['date', 'exdate'])
    else:
        prs_dataset = pd.read_csv(dataset_path, parse_dates=['date', 'exdate'])
    return prs_dataset
def read_calibrated_result(calibrated_result_path = '../Benchmarks/Benchmarks_afd0V_afconstr/MSE_100MAPE/20220906_involveCalibParam_MSE100MAPE/bs_involveCalibParam_MSE100MAPE/bs_calibrated_dataset.csv'):
    if os.path.isdir(calibrated_result_path):
        clb_dataset = dd.read_csv(calibrated_result_path + "/*", parse_dates=['date'])
    else:
        clb_dataset = pd.read_csv(calibrated_result_path, parse_dates=['date'])
    return clb_dataset

def calc_date_list(whole_date_list, start_date='20100101', end_date='20160531'):
    date_list = np.sort(np.unique(whole_date_list))
    after_start_date = date_list >= np.datetime64(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}")
    if np.argmax(after_start_date) != 0:
        after_start_date[np.argmax(after_start_date)-1] = True # is start date as test date, not start date as train date
    before_end_date = date_list <= np.datetime64(f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}")
    if np.argmin(before_end_date) != 1:
        before_end_date[np.argmin(before_end_date)-2] = True
    date_list = date_list[after_start_date & before_end_date]
    return date_list

def calibrate(N = 2 ** 15, eta = 0.01, alpha = 1, window_size = 1, dataset_path = "../Dataset/prs_dataset.csv", calibrated_result_path='../Benchmarks/Benchmarks_afd0V_afconstr/MSE_100MAPE/20220906_involveCalibParam_MSE100MAPE/bs_involveCalibParam_MSE100MAPE/bs_calibrated_dataset.csv', eval_on=False, key_metrics=('date', 'calib_params', 'train_mse', 'train_mape', 'test_mse', 'test_mape'), result_folder = '../Benchmarks', mode='bs', start_date='19960101', end_date='20160531'):
    prs_dataset = read_dataset(dataset_path)
    clb_dataset = None
    if eval_on:
        clb_dataset = read_calibrated_result(calibrated_result_path)
    date_list = calc_date_list(prs_dataset['date'], start_date, end_date)
    print(f"process test date from {str(date_list[window_size])[:10]} to {str(date_list[-1])[:10]}")

    result_table = pd.DataFrame(columns=key_metrics)

    for idx, train_date in enumerate(date_list[:-window_size]):

        test_date = date_list[idx+window_size]
        performance = {k:None for k in key_metrics}

        # train & test
        train_dataset = compute_func(prs_dataset[prs_dataset['date'] == train_date].reset_index(drop=True))
        test_dataset = compute_func(prs_dataset[prs_dataset['date'] == test_date].reset_index(drop=True))

        # get calibrated parameters
        if eval_on:
            record_metric = clb_dataset[clb_dataset['date'] == test_date]
            calibrated_params = record_metric['calib_params'].item()
            calibrated_params = np.array([float(x) for x in calibrated_params[1:-1].split(',')])
        else:
            calibrated_params = optimization(N, eta, alpha, train_dataset, mode)

        # record calibrated parameters and loss
        performance['date'] = str(test_date)[:10]
        performance['calib_params'] = str(list(calibrated_params))
        for key in key_metrics:
            if key == 'date' or key == 'calib_params':
                continue

            ltdf = train_dataset if 'train' in key else test_dataset
            invm = ltdf['strike_price'] / ltdf['S']
            if 'deepitm' in key.lower():
                mask = filter(invm, 'i', invm.mean() - 2 * invm.std())
            elif 'deepotm' in key.lower():
                mask = filter(invm, 'o', invm.mean() + 2 * invm.std())
            elif 'itm' in key.lower():
                mask = filter(invm, 'i', 1)
            elif 'otm' in key.lower():
                mask = filter(invm, 'o', 1)
            elif 'atm' in key.lower():
                atm_itm = filter(invm, 'o', invm.mean() - 1*invm.std()) & filter(invm, 'i')
                atm_otm = filter(invm, 'i', invm.mean() + 1*invm.std()) & filter(invm, 'o')
                mask = atm_itm | atm_otm
            else:
                mask = pd.Series([True] * len(ltdf))
            mse, mape = general_loss(N, eta, alpha, ltdf[mask], mode, *calibrated_params)
            performance[key] = mse if 'mse' in key.lower() else mape

        # if 'train_mse' in key_metrics or 'train_mape' in key_metrics:
        #     train_MSE, train_MAPE = general_loss(N, eta, alpha, train_dataset, mode, *calibrated_params)
        #     performance['train_mse'] = train_MSE
        #     performance['train_mape'] = train_MAPE
        # if 'test_mse' in key_metrics or 'test_mape' in key_metrics:
        #     test_MSE, test_MAPE = general_loss(N, eta, alpha, test_dataset, mode, *calibrated_params)
        #     performance['test_mse'] = test_MSE
        #     performance['test_mape'] = test_MAPE

        message = f"date={str(test_date)[:10]}, calib_params={calibrated_params}"
        message += ''.join([f", {key}={performance[key]:.2f}" for key in key_metrics if key != 'date' and key != 'calib_params'])
        print(message)

        result_table = pd.concat([result_table, pd.DataFrame.from_dict(performance, orient='index').T], ignore_index=True)
        if idx % 10 == 9:
            # update to prevent unexpected accident
            result_table.to_csv(f"{result_folder}/date_loss_message_dataframe.csv", index=False)

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=90)) # one quarter is about 90 days
            plt.plot([np.datetime64(x) for x in result_table['date']], result_table['test_mape'])
            plt.xlabel('date')
            plt.ylabel('test_mape')
            plt.gcf().autofmt_xdate()
            plt.savefig(f"{result_folder}/{mode}quarter_loss_curve_test_mape.png")
            plt.show()
            plt.close()
    result_table.to_csv(f"{result_folder}/{mode}date_loss_message_dataframe.csv", index=False)
    return result_table

if __name__ == '__main__':
    # 設定參數
    N = 2 ** 15
    eta = 0.01
    alpha = 1
    window_size = 1
    mode = 'kj'
    dataset_path = "../data/prs_dataset_new.csv"
    calibrated_result_path = f'/data/{mode}_calibrated_dataset.csv'
    eval_on = False
    key_metrics = ('date', 'calib_params', 'train_mse', 'train_mape', 'test_mse', 'test_mape')
    result_folder = 'result'
    start_date = '20210101'
    end_date = '20221231'
    
    result_table = calibrate(N, eta, alpha, window_size, dataset_path, calibrated_result_path, 
                             eval_on, key_metrics, result_folder, mode, start_date, end_date)
    
    print(result_table)