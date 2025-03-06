import pandas as pd
import numpy as np
from functions import *
import os
from datetime import datetime,timedelta
from tqdm import tqdm

def data_preprocess():
    # 文件路径
    input_file_path = r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/TXO_with_TXF.csv"

    # filled time series data
    data = pd.read_csv(input_file_path)
    data['r'] = data['r']/100
    # 去除 PC 列中的前後空格
    data['PC'] = data['PC'].str.strip()
    data['date'] = pd.to_datetime(data['date'])
    data['exdate'] = pd.to_datetime(data['exdate'])
    # specificed data
    data = data[data['date'] >= '2021-01-01']

    # fill missing data
    data = fill_missing_data(data)
    data.to_csv(f'filled_time_series_underlying_data.csv')

    return data

def find_missing_dates(option_data, trading_dates):
    start_date = option_data.iloc[0]['date']
    end_date = option_data.iloc[0]['exdate']
    
    option_trading_dates = trading_dates[(trading_dates >= start_date) & (trading_dates <= end_date)]
    existing_dates = set(option_data['date'].unique())

    # missing dates
    missing_dates = sorted(set(option_trading_dates) - existing_dates)

    return missing_dates
    
def get_underlying(data, missing_date, exdate):
    if not isinstance(missing_date, pd.Timestamp):
        missing_date = pd.to_datetime(missing_date)

    if not isinstance(exdate, pd.Timestamp):
        exdate = pd.to_datetime(exdate)
        
    temp_data = data[(data['date'] == missing_date) & (data['exdate'] == exdate)]
    
    if not temp_data.empty:
        return temp_data['S'].iloc[0]  
    else:
        return np.nan
        '''
        all_exdates = data[data['date'] == missing_date]['exdate'].unique()
        if len(all_exdates) > 0:

            # TODO : 1. 因為 exdate 相差太多(不同月份)，考慮以前一天的 underlying 去做填補補
            #        2. 若以前一天的 underlying 去做填補則去做浮動調整
            
            date_diffs = [abs((exdate - ed).days) for ed in all_exdates]
            nearest_exdate = all_exdates[np.argmin(date_diffs)]
            nearest_data = data[(data['date'] == missing_date) & (data['exdate'] == nearest_exdate)]
            if not nearest_data.empty:
                return nearest_data['S'].iloc[0]
        else:
            return np.nan  
        '''
    
def fill_missing_data(data):
    trading_dates = data['date'].unique()
    option_groups = data.groupby(['strike_price', 'exdate', 'PC'])
    
    # 創建一個 tqdm 進度條
    progress_bar = tqdm(total=len(option_groups), desc="Filling missing data")
    
    for Opt_ID, option_data in option_groups:
        option_data = option_data.sort_values('date')  
        missing_dates = find_missing_dates(option_data, trading_dates)
        
        for missing_date in missing_dates:
            S0 = option_data['strike_price'].iloc[0]
            exdate = option_data['exdate'].iloc[0]
            r = option_data['r'].iloc[0]
            d = option_data['d'].iloc[0]
            cp_flag = option_data['PC'].iloc[0]
            price = 0
            tau = 0

            # find spot price
            spot_price = get_underlying(data, missing_date, exdate)
            
            new_record = option_data.iloc[0].copy()
            new_record['date'] = missing_date
            new_record['option_price'] = price
            new_record['tau'] = tau
            new_record['invm'] = S0 / spot_price
            new_record['S'] = spot_price
            new_record['volume'] = 0

            # print(f'Missing data : {missing_date}, New record : {new_record}')
            data = pd.concat([data, pd.DataFrame([new_record])], ignore_index=True)
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    return data

# preprocess data
data = data_preprocess()