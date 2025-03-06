import pandas as pd
import numpy as np
import torch
from tqdm import trange
from functions import *
import os
from datetime import datetime,timedelta
from tqdm import tqdm
from benchmarks_fftoption import compute_call_price_by_parametric_model

def data_preprocess():
    # 文件路径
    input_file_path = r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/TXO_with_TXF.csv"
    output_file_path = r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/prs_dataset_new.csv"

    # 检查文件是否存在
    if os.path.exists(output_file_path):
        # 如果文件存在，直接读取
        data = pd.read_csv(output_file_path)
        print("Dataset loaded from prs_dataset_new.csv")

    else:
        # filled time series data
        # data = pd.read_csv(input_file_path)
        # data['r'] = data['r']/100
        # # 去除 PC 列中的前後空格
        # data['PC'] = data['PC'].str.strip()
        # data['date'] = pd.to_datetime(data['date'])
        # data['exdate'] = pd.to_datetime(data['exdate'])
        # # specificed data
        # data = data[data['date'] >= '2021-01-01']

        # # fill missing data
        # data = fill_missing_data(data)
        # data.to_csv(f'filled_time_series_data.csv')

        # read time series data
        data = pd.read_csv(f'filled_time_series_data.csv')
        data['PC'] = data['PC'].str.strip()
        data['date'] = pd.to_datetime(data['date'])
        data['exdate'] = pd.to_datetime(data['exdate'])

        # 使用 spline interpolate
        data = spline_interpolate(data)
        
        # 計算相關值
        data['impl_volatility'] = data.apply(calculate_implied_volatility, axis = 1)
        data['delta'] = data.apply(BSM_delta, axis = 1)
        data['gamma'] = data.apply(BSM_gamma, axis = 1)
        data['theta'] = data.apply(BSM_theta, axis = 1)
        data['rho'] = data.apply(BSM_rho, axis = 1)
        data['vega'] = data.apply(BSM_vega, axis = 1)
        data['theory_price'] = data.apply(calculate_theory_price, axis = 1)

        data['opt_ID'] = data.groupby(['strike_price', 'exdate', 'PC']).ngroup()
        data['pre_settle_price'] = data.groupby('opt_ID')['option_price'].shift(1)
        data['settle_price_chg'] = ((data['option_price'] - data['pre_settle_price']) / data['pre_settle_price']) * 100
        data['theory_margin'] = data['theory_price'] - data['option_price']
        data['PC'] = data['PC'].map({'C': 0, 'P': 1})
        data = data.dropna().reset_index(drop=True)
        data.to_csv(output_file_path)
        print("Dataset preprocessed and saved to prs_dataset.csv")
    
    return data

def find_missing_dates(option_data, trading_dates):
    start_date = option_data.iloc[0]['date']
    end_date = option_data.iloc[0]['exdate']
    
    option_trading_dates = trading_dates[(trading_dates >= start_date) & (trading_dates <= end_date)]
    existing_dates = set(option_data['date'].unique())

    # missing dates
    missing_dates = sorted(set(option_trading_dates) - existing_dates)

    return missing_dates

def fetch_data_for_missing_date(missing_date, strike_price, exdate, cp_flag):
    missing_date = str(missing_date)[:10]
    exdate = str(exdate)[:10]
    missing_date = datetime.strptime(missing_date, "%Y-%m-%d").date()
    exdate = datetime.strptime(exdate, "%Y-%m-%d").date()
    year = missing_date.year
    month = missing_date.month
    
    file_path = f"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/{year}_opt_ajusted/{year}_opt_{month:02d}.csv"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)  
        df['date'] = pd.to_datetime(df['date'], format= "%Y-%m-%d").dt.date
        df['exdate'] = pd.to_datetime(df['exdate'], format= "%Y-%m-%d").dt.date

        # 使用 strike price 和 exdate 獲取特定數據
        specific_data = df[(df['date'] == missing_date) & (df['strike_price'] == strike_price) & (df['exdate'] == exdate) & (df['cp_flag'] == cp_flag)]
        
        if not specific_data.empty:
            return specific_data
        else:
            print(f"No data found for strike price {strike_price} and exdate {exdate} on {missing_date}")
            return None
    else:
        print(f"File not found for date {missing_date}")

def get_tau(data, missing_date, exdate):
    if not isinstance(missing_date, pd.Timestamp):
        missing_date = pd.to_datetime(missing_date)

    if not isinstance(exdate, pd.Timestamp):
        exdate = pd.to_datetime(exdate)
        
    temp_data = data[(data['date'] == missing_date) & (data['exdate'] == exdate)]
    
    if not temp_data.empty:
        return temp_data['tau'].iloc[0]
    else:
        trading_days = pd.bdate_range(start=missing_date, end=exdate, freq='B')
        n_trading_days = len(trading_days)
        tau = n_trading_days / 252    
        return tau
    
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
            
            complementary_data = fetch_data_for_missing_date(missing_date, S0, exdate, cp_flag)
            if complementary_data is not None:
                price = complementary_data['option_price'].iloc[0] 
            else:
                price = 0
            
            # find spot price
            spot_price = get_underlying(data, missing_date, exdate)

            # find tau
            tau = get_tau(data, missing_date, exdate)
            
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

def spline_interpolate(data):
    data['date'] = pd.to_datetime(data['date'])
    data['exdate'] = pd.to_datetime(data['exdate'])
    # construct lookup table
    lookup_df = (data.groupby(['date', 'exdate'])['S']
                    .first()
                    .reset_index()
                    .pivot(index='date', columns='exdate', values='S'))
    
    # use spline interpolate
    for col in lookup_df.columns:
        lookup_df[col] = lookup_df[col].interpolate(
            method='linear',
            limit_direction='both',
            limit_area='inside'  # 只填補中間值
        ).apply(lambda x: '{:.2f}'.format(x)).astype(float)
    
    # filling underlying price
    lookup_dict = (lookup_df.melt(ignore_index=False)
                            .reset_index()
                            .set_index(['date','exdate'])['value']
                            .to_dict())
    
    mask = data['S'].isna()
    data.loc[mask, 'S'] = data[mask].apply(
        lambda row: lookup_dict.get((row['date'], row['exdate'])), 
        axis=1
    )

    # adjust invm
    data.loc[mask, 'invm'] = data[mask].apply(
        lambda row: row['strike_price']/row['S'], 
        axis=1
    )
    
    return data

# preprocess data
# data = data_preprocess()

data = pd.read_csv(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/prs_dataset_new.csv")

# 將一個標的的時間序列分割 train valid input，並記錄他們的timestamp
data = data[data['date'] >= '2021-01-01']
data['date'] = pd.to_datetime(data['date'])
train_input = []
train_label = []
valid_input = []
valid_label = []
test_input = []
test_label = []
train_label_timestamp = []
valid_label_timestamp = []
test_label_timestamp = []
seq_len = 10
train_days = 3
valid_days = 1
test_days = 1

def build_input_label(data, index, seq_len):
    input = torch.zeros((3, seq_len, 5))
    tmp_input = data.iloc[index:index+seq_len+1]
    if len(tmp_input) < seq_len + 1:  # Check if we have enough data
        return None, None, None
        
    input[0, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['volume','PC','tau','impl_volatility','strike_price']], dtype=np.float64))
    input[1, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['delta','gamma','rho','theta','vega']], dtype=np.float64))
    input[2, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['pre_settle_price','settle_price_chg','S','theory_margin','theory_price']], dtype=np.float64))
    label = torch.tensor(np.array(tmp_input.iloc[seq_len]['option_price'], dtype=np.float64))
    date_tensor = torch.tensor(np.array([tmp_input.iloc[seq_len]['date'].year, 
                                       tmp_input.iloc[seq_len]['date'].month, 
                                       tmp_input.iloc[seq_len]['date'].day], dtype=np.int64))
    return input, label, date_tensor

# Sort dates and create non-overlapping windows
unique_dates = sorted(data['date'].unique())
total_window = train_days + valid_days + test_days + seq_len  # Include sequence length in window
step_size = total_window  # Use full window size as step to avoid overlap

for i in tqdm(range(0, len(unique_dates) - total_window + 1, step_size)):
    # Define date ranges for each set
    window_dates = unique_dates[i:i+total_window]
    
    train_dates = window_dates[:train_days+seq_len]
    valid_dates = window_dates[train_days:train_days+valid_days+seq_len]
    test_dates = window_dates[train_days+valid_days:train_days+valid_days+test_days+seq_len]
    
    for opt_id in data['opt_ID'].unique():
        opt_data = data[data['opt_ID'] == opt_id].sort_values('date')
        
        # Get data for each set
        train_data = opt_data[opt_data['date'].isin(train_dates)]
        valid_data = opt_data[opt_data['date'].isin(valid_dates)]
        test_data = opt_data[opt_data['date'].isin(test_dates)]
        
        # Process training data
        if len(train_data) >= seq_len + 1:
            for j in range(len(train_data) - seq_len):
                input, label, date_tensor = build_input_label(train_data, j, seq_len)
                if input is not None:
                    train_input.append(input)
                    train_label.append(label)
                    train_label_timestamp.append(date_tensor)
        
        # Process validation data
        if len(valid_data) >= seq_len + 1:
            for j in range(len(valid_data) - seq_len):
                input, label, date_tensor = build_input_label(valid_data, j, seq_len)
                if input is not None:
                    valid_input.append(input)
                    valid_label.append(label)
                    valid_label_timestamp.append(date_tensor)
        
        # Process test data
        if len(test_data) >= seq_len + 1:
            for j in range(len(test_data) - seq_len):
                input, label, date_tensor = build_input_label(test_data, j, seq_len)
                if input is not None:
                    test_input.append(input)
                    test_label.append(label)
                    test_label_timestamp.append(date_tensor)


# for i in trange(data['opt_ID'].unique().shape[0]):
#     index_len = data.loc[(data['opt_ID'] == data['opt_ID'].unique()[i])].shape[0] - seq_len
#     if (index_len) >= 0:
#         index_arr = np.arange(index_len)
#         train_index_arr = index_arr[:int(0.8*index_len)]
#         valid_index_arr = index_arr[int(0.8*index_len):int(0.9*index_len)]
#         test_index_arr = index_arr[int(0.9*index_len):]

#         for index in train_index_arr:
#             input = torch.zeros((3, 10, 5))
#             # 分為三個 channel 
#             tmp_input = data.loc[(data['opt_ID'] == data['opt_ID'].unique()[i])][index:index+seq_len+1]
#             input[0, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['volume','PC','tau','impl_volatility','strike_price']], dtype=np.float64))
#             # 基本面 inventory   call(label 0 )/put(label 1 )   tau(days)   impli_vol   strikePrice
#             input[1, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['delta','gamma','rho','theta','vega']], dtype=np.float64))
#             # 希臘字母 delta   gamma   rho   theta   vega
#             input[2, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['pre_settle_price','settle_price_chg','S','theory_margin','theory_price']], dtype=np.float64))
#             # 價格 pre_settlePrice   settlePrice_chg   spotPrice   theoryMargin   theoryPrice
#             label = torch.tensor(np.array(tmp_input.iloc[seq_len]['option_price'], dtype=np.float64))
#             # 將日期字符串轉換為datetime對象
#             date = datetime.strptime(tmp_input.iloc[seq_len]['date'], "%Y-%m-%d")
#             # 從datetime對象中提取年、月、日
#             year = date.year
#             month = date.month
#             day = date.day
#             date_tensor = torch.tensor(np.array([year, month, day], dtype=np.int64))
#             train_label_timestamp.append(date_tensor)
#             train_input.append(input)
#             train_label.append(label)

#         for index in valid_index_arr:
#             input = torch.zeros((3, 10, 5))
#             # 分為三個 channel 
#             tmp_input = data.loc[(data['opt_ID'] == data['opt_ID'].unique()[i])][index:index+seq_len+1]
#             input[0, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['volume','PC','tau','impl_volatility','strike_price']], dtype=np.float64))
#             # 基本面 inventory   call(label 0 )/put(label 1 )   tau(days)   impli_vol   strikePrice
#             input[1, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['delta','gamma','rho','theta','vega']], dtype=np.float64))
#             # 希臘字母 delta   gamma   rho   theta   vega
#             input[2, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['pre_settle_price','settle_price_chg','S','theory_margin','theory_price']], dtype=np.float64))
#             # 價格 pre_settlePrice   settlePrice_chg   spotPrice   theoryMargin   theoryPrice
#             label = torch.tensor(np.array(tmp_input.iloc[seq_len]['option_price'], dtype=np.float64))
#             # 將日期字符串轉換為datetime對象
#             date = datetime.strptime(tmp_input.iloc[seq_len]['date'], "%Y-%m-%d")
#             # 從datetime對象中提取年、月、日
#             year = date.year
#             month = date.month
#             day = date.day
#             date_tensor = torch.tensor(np.array([year, month, day], dtype=np.int64))
#             valid_label_timestamp.append(date_tensor)
#             valid_input.append(input)
#             valid_label.append(label)

#         for index in test_index_arr:
#             input = torch.zeros((3, 10, 5))
#             # 分為三個 channel 
#             tmp_input = data.loc[(data['opt_ID'] == data['opt_ID'].unique()[i])][index:index+seq_len+1]
#             input[0, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['volume','PC','tau','impl_volatility','strike_price']], dtype=np.float64))
#             # 基本面 inventory   call(label 0 )/put(label 1 )   tau(days)   impli_vol   strikePrice
#             input[1, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['delta','gamma','rho','theta','vega']], dtype=np.float64))
#             # 希臘字母 delta   gamma   rho   theta   vega
#             input[2, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['pre_settle_price','settle_price_chg','S','theory_margin','theory_price']], dtype=np.float64))
#             # 價格 pre_settlePrice   settlePrice_chg   spotPrice   theoryMargin   theoryPrice
#             label = torch.tensor(np.array(tmp_input.iloc[seq_len]['option_price'], dtype=np.float64))
#             # 將日期字符串轉換為datetime對象
#             date = datetime.strptime(tmp_input.iloc[seq_len]['date'], "%Y-%m-%d")
#             # 從datetime對象中提取年、月、日
#             year = date.year
#             month = date.month
#             day = date.day
#             date_tensor = torch.tensor(np.array([year, month, day], dtype=np.int64))
#             test_label_timestamp.append(date_tensor)
#             test_input.append(input)
#             test_label.append(label)

# Verify no overlap between sets
def check_timestamp_overlap(train_timestamps, valid_timestamps, test_timestamps):
    train_dates = set((int(t[0]), int(t[1]), int(t[2])) for t in train_timestamps)
    valid_dates = set((int(t[0]), int(t[1]), int(t[2])) for t in valid_timestamps)
    test_dates = set((int(t[0]), int(t[1]), int(t[2])) for t in test_timestamps)
    
    train_valid_overlap = train_dates.intersection(valid_dates)
    train_test_overlap = train_dates.intersection(test_dates)
    valid_test_overlap = valid_dates.intersection(test_dates)
    
    if train_valid_overlap or train_test_overlap or valid_test_overlap:
        print("WARNING: Found overlapping dates between sets!")
        if train_valid_overlap:
            print(f"Train-Valid overlap: {len(train_valid_overlap)} dates")
        if train_test_overlap:
            print(f"Train-Test overlap: {len(train_test_overlap)} dates")
        if valid_test_overlap:
            print(f"Valid-Test overlap: {len(valid_test_overlap)} dates")
    else:
        print("No overlap found between train, validation and test sets")

# Convert lists to tensors
def convert_to_tensors(input_list, label_list, timestamp_list, name):
    if not input_list or not label_list or not timestamp_list:
        print(f"WARNING: Empty lists for {name} set")
        return None, None, None
        
    input_tensor = torch.stack(input_list)
    label_tensor = torch.stack(label_list)
    timestamp_tensor = torch.stack(timestamp_list)
    
    print(f"Shape of the {name} Input:", input_tensor.shape)
    print(f"Shape of the {name} Label:", label_tensor.shape)
    print(f"Shape of the {name} Label Timestamp:", timestamp_tensor.shape)
    
    return input_tensor, label_tensor, timestamp_tensor

# Check for overlaps
check_timestamp_overlap(train_label_timestamp, valid_label_timestamp, test_label_timestamp)

# Convert to tensors
train_input, train_label, train_label_timestamp = convert_to_tensors(
    train_input, train_label, train_label_timestamp, "Training")

valid_input, valid_label, valid_label_timestamp = convert_to_tensors(
    valid_input, valid_label, valid_label_timestamp, "Validating")

test_input, test_label, test_label_timestamp = convert_to_tensors(
    test_input, test_label, test_label_timestamp, "Testing")

# Verify data was converted successfully
if train_input is None or valid_input is None or test_input is None:
    print("ERROR: Failed to convert some datasets to tensors")


# Shape of the Training Input: torch.Size([12982, 3, 10, 5])
# Shape of the Training Label: torch.Size([12982])
# Shape of the Training Label Timestamp: torch.Size([12982, 3])
# Shape of the Validating Input: torch.Size([4036, 3, 10, 5])
# Shape of the Validating Label: torch.Size([4036])
# Shape of the Validating Label Timestamp: torch.Size([4036, 3])
# Shape of the Testing Input: torch.Size([3846, 3, 10, 5])
# Shape of the Testing Label: torch.Size([3846])
# Shape of the Testing Label Timestamp: torch.Size([3846, 3])


torch.save(train_input, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_input_taiex2_syn.pt")
torch.save(train_label, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_label_taiex2_syn.pt")
torch.save(train_label_timestamp, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_label_timestamp_taiex2_syn.pt")
torch.save(test_input, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_input_taiex2_syn.pt")
torch.save(test_label, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_label_taiex2_syn.pt")
torch.save(test_label_timestamp, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_label_timestamp_taiex2_syn.pt")
torch.save(valid_input, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_input_taiex2_syn.pt")
torch.save(valid_label, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_label_taiex2_syn.pt")
torch.save(valid_label_timestamp, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_label_timestamp_taiex2_syn.pt")
