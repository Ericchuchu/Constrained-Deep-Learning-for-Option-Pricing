import pandas as pd
import numpy as np
import torch
import os
from tqdm import trange, tqdm
from functions import *
from math import log,sqrt

def calculate_date_position(dates, base_date=None):
    """Calculate relative position of dates from base date"""
    if base_date is None:
        base_date = min(dates)
    return [(date - base_date).days for date in dates]

def data_preprocess():
    output_file_path = r"/Users/chuchu/Desktop/option_pricing_thesis_project/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/prs_dataset_mpf.csv"

    # Check if preprocessed file exists
    if os.path.exists(output_file_path):
        data = pd.read_csv(output_file_path)
        data['date'] = pd.to_datetime(data['date'])
        data['exdate'] = pd.to_datetime(data['exdate'])
        print("Dataset loaded from prs_dataset_mpf.csv")
    else:
        # Read time series data
        data = pd.read_csv('filled_time_series_data.csv')
        data['PC'] = data['PC'].str.strip()
        data['date'] = pd.to_datetime(data['date'])
        data['exdate'] = pd.to_datetime(data['exdate'])

        # Use spline interpolate
        data = spline_interpolate(data)
        
        # Calculate related values
        data['impl_volatility'] = data.apply(calculate_implied_volatility, axis=1)
        data['delta'] = data.apply(BSM_delta, axis=1)
        data['gamma'] = data.apply(BSM_gamma, axis=1)
        data['theta'] = data.apply(BSM_theta, axis=1)
        data['rho'] = data.apply(BSM_rho, axis=1)
        data['vega'] = data.apply(BSM_vega, axis=1)
        data['theory_price'] = data.apply(calculate_theory_price, axis=1)

        # moneyness based on option type
        data['moneyness'] = data.apply(
            lambda row: -log(row['invm'])/(row['impl_volatility']*sqrt(row['tau'])) if row['PC'] == 'C' 
            else log(row['invm'])/(row['impl_volatility']*sqrt(row['tau'])), 
            axis=1
        )

        data['opt_ID'] = data.groupby(['strike_price', 'exdate', 'PC']).ngroup()
        data['pre_settle_price'] = data.groupby('opt_ID')['option_price'].shift(1)
        data['settle_price_chg'] = ((data['option_price'] - data['pre_settle_price']) / data['pre_settle_price']) * 100
        data['theory_margin'] = (data['theory_price'] - data['option_price']) / data['option_price']
   
        # Convert PC to numeric
        data['PC'] = data['PC'].map({'C': 0, 'P': 1})
        
        data = data.dropna().reset_index(drop=True)

        # apply log scaling
        epsilon = 1e-10  # 或其他適合您數據範圍的小常數
        data['option_price'] = np.log(data['option_price'] + epsilon)
        data['pre_settle_price'] = np.log(data['pre_settle_price'] + epsilon)
        data['theory_price'] = np.log(data['theory_price'] + epsilon)
        data['strike_price'] = np.log(data['strike_price'] + epsilon)
        data['S'] = np.log(data['S']+ epsilon)

        data.to_csv(output_file_path)
        print("Dataset preprocessed and saved to prs_dataset_mpf.csv")
    
    # Calculate date positions
    base_date = data['date'].min()
    data['date_position'] = calculate_date_position(data['date'], base_date)
    
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

def build_input_label(data, index, seq_len):
    input = torch.zeros((3, seq_len, 5))

    tmp_input = data.iloc[index:index+seq_len+1]
    if len(tmp_input) < seq_len + 1:
        return None, None, None, None
        
    # ConvLSTM input
    input[0, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['volume','PC','tau','impl_volatility','moneyness']], dtype=np.float64))
    input[1, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['delta','gamma','rho','theta','vega']], dtype=np.float64))
    input[2, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['option_price','settle_price_chg','S','theory_margin','theory_price']], dtype=np.float64))
    
    label = torch.tensor(np.array(tmp_input.iloc[seq_len]['option_price'], dtype=np.float64))
    date_tensor = torch.tensor(np.array([tmp_input.iloc[seq_len]['date'].year, 
                                       tmp_input.iloc[seq_len]['date'].month, 
                                       tmp_input.iloc[seq_len]['date'].day], dtype=np.int64))
    
    return input, label, date_tensor


def prepare_data(data, seq_len=10, train_days=3, valid_days=1, test_days=1):
    train_input = []
    train_label = []
    valid_input = []
    valid_label = []
    test_input= []
    test_label = []
    train_label_timestamp = []
    valid_label_timestamp = []
    test_label_timestamp = []

    # Sort dates and create non-overlapping windows
    unique_dates = sorted(data['date'].unique())
    total_window = train_days + valid_days + test_days + seq_len
    step_size = total_window

    for i in tqdm(range(0, len(unique_dates) - total_window + 1, step_size)):
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

    # Convert lists to tensors
    train_input = torch.stack(train_input)
    train_label = torch.stack(train_label)
    train_label_timestamp = torch.stack(train_label_timestamp)

    valid_input = torch.stack(valid_input)
    valid_label = torch.stack(valid_label)
    valid_label_timestamp = torch.stack(valid_label_timestamp)

    test_input = torch.stack(test_input)
    test_label = torch.stack(test_label)
    test_label_timestamp = torch.stack(test_label_timestamp)

    # Save tensors
    save_dir = "/Users/chuchu/Desktop/option_pricing_thesis_project/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main//data/torch-data"
    torch.save(train_input, f"{save_dir}/train_input_mpf.pt")
    torch.save(train_label, f"{save_dir}/train_label_mpf.pt")
    torch.save(train_label_timestamp, f"{save_dir}/train_label_timestamp_mpf.pt")
    
    torch.save(valid_input, f"{save_dir}/valid_input_mpf.pt")
    torch.save(valid_label, f"{save_dir}/valid_label_mpf.pt")
    torch.save(valid_label_timestamp, f"{save_dir}/valid_label_timestamp_mpf.pt")
    
    torch.save(test_input, f"{save_dir}/test_input_mpf.pt")
    torch.save(test_label, f"{save_dir}/test_label_mpf.pt")
    torch.save(test_label_timestamp, f"{save_dir}/test_label_timestamp_mpf.pt")

    return (train_input, train_label, train_label_timestamp,
            valid_input, valid_label, valid_label_timestamp,
            test_input, test_label, test_label_timestamp)

def prepare_rolling_window_data(data, seq_len=10, initial_train_days=3, valid_days=1, test_days=1, subsequent_train_days=5):
    """
    累積滾動策略：
    1. 第一個窗口：使用前 initial_train_days 天訓練，接著的 valid_days 天驗證，再接著 test_days 天測試
    2. 第二個窗口：將先前的(訓練+驗證+測試)資料全部納入訓練集，總計 subsequent_train_days 天訓練，
       然後使用接下來的 valid_days 天驗證，再接著 test_days 天測試
    3. 依此類推
    
    Args:
        data: 輸入的資料 DataFrame
        seq_len: 序列長度
        initial_train_days: 第一個窗口的訓練天數
        valid_days: 每個窗口的驗證天數
        test_days: 每個窗口的測試天數
        subsequent_train_days: 後續窗口的訓練天數
    
    Returns:
        包含每個滾動窗口的訓練、驗證、測試資料的字典列表
    """
    # 排序日期
    unique_dates = sorted(data['date'].unique())
    
    all_windows = []
    current_position = 0
    
    window_idx = 0
    while True:
        print(f"處理窗口 {window_idx}...")
        
        # 計算當前窗口的範圍
        if window_idx == 0:
            # 第一個窗口
            train_start = current_position
            train_end = train_start + initial_train_days
            required_days = initial_train_days + valid_days + test_days
        else:
            # 後續窗口
            train_start = train_end 
            train_end = current_position + initial_train_days  # 包含先前的所有數據
            required_days = subsequent_train_days + valid_days + test_days
        
        valid_start = train_end
        valid_end = valid_start + valid_days
        test_start = valid_end
        test_end = test_start + test_days
        
        # 檢查是否有足夠的日期
        if test_end+seq_len > len(unique_dates):
            print(f"沒有足夠的日期資料來創建窗口 {window_idx}，停止處理")
            break
        
        # 獲取日期範圍
        train_dates = unique_dates[train_start:train_end+seq_len]
        valid_dates = unique_dates[valid_start:valid_end+seq_len]
        test_dates = unique_dates[test_start:test_end+seq_len]
        
        # 為下一個窗口更新位置
        current_position = test_start  # 下一個窗口從當前測試集的開始位置開始
        
        # 初始化該窗口的資料容器
        window_data = {
            'window_idx': window_idx,
            'train_start_date': train_dates[0],
            'train_end_date': train_dates[-1],
            'valid_start_date': valid_dates[0],
            'valid_end_date': valid_dates[-1],
            'test_start_date': test_dates[0],
            'test_end_date': test_dates[-1],
            'train_input': [],
            'train_label': [],
            'train_timestamp': [],
            'valid_input': [],
            'valid_label': [],
            'valid_timestamp': [],
            'test_input': [],
            'test_label': [],
            'test_timestamp': []
        }
        
        # 為每個選擇權 ID 處理資料
        for opt_id in data['opt_ID'].unique():
            opt_data = data[data['opt_ID'] == opt_id].sort_values('date')
            
             # Get data for each set
            train_data = opt_data[opt_data['date'].isin(train_dates)]
            valid_data = opt_data[opt_data['date'].isin(valid_dates)]
            test_data = opt_data[opt_data['date'].isin(test_dates)]
            
            # 處理訓練資料
            if len(train_data) >= seq_len + 1:
                for j in range(len(train_data) - seq_len):
                    input_tensor, label_tensor, date_tensor = build_input_label(train_data, j, seq_len)
                    if input_tensor is not None:
                        window_data['train_input'].append(input_tensor)
                        window_data['train_label'].append(label_tensor)
                        window_data['train_timestamp'].append(date_tensor)
            
            # 處理驗證資料
            if len(valid_data) >= seq_len + 1:
                for j in range(len(valid_data)-seq_len):
                    input_tensor, label_tensor, date_tensor = build_input_label(valid_data, j, seq_len)
                    if input_tensor is not None:
                        window_data['valid_input'].append(input_tensor)
                        window_data['valid_label'].append(label_tensor)
                        window_data['valid_timestamp'].append(date_tensor)
            
            # 處理測試資料
            if len(test_data) >= seq_len + 1:
                for j in range(len(test_data) - seq_len):
                    input_tensor, label_tensor, date_tensor = build_input_label(test_data, j, seq_len)
                    if input_tensor is not None:
                        window_data['test_input'].append(input_tensor)
                        window_data['test_label'].append(label_tensor)
                        window_data['test_timestamp'].append(date_tensor)
        
        # 檢查是否有足夠的資料
        if (len(window_data['train_input']) > 0 and 
            len(window_data['valid_input']) > 0 and 
            len(window_data['test_input']) > 0):
            
            # 轉換為張量
            window_data['train_input'] = torch.stack(window_data['train_input'])
            window_data['train_label'] = torch.stack(window_data['train_label'])
            window_data['train_timestamp'] = torch.stack(window_data['train_timestamp'])
            
            window_data['valid_input'] = torch.stack(window_data['valid_input'])
            window_data['valid_label'] = torch.stack(window_data['valid_label'])
            window_data['valid_timestamp'] = torch.stack(window_data['valid_timestamp'])
            
            window_data['test_input'] = torch.stack(window_data['test_input'])
            window_data['test_label'] = torch.stack(window_data['test_label'])
            window_data['test_timestamp'] = torch.stack(window_data['test_timestamp'])
            
            # 添加到結果列表
            all_windows.append(window_data)
            
            print(f"窗口 {window_idx}: "
                  f"訓練: {window_data['train_start_date']} 到 {window_data['train_end_date']} "
                  f"({len(window_data['train_input'])} 樣本), "
                  f"驗證: {window_data['valid_start_date']} 到 {window_data['valid_end_date']} "
                  f"({len(window_data['valid_input'])} 樣本), "
                  f"測試: {window_data['test_start_date']} 到 {window_data['test_end_date']} "
                  f"({len(window_data['test_input'])} 樣本)")
        else:
            print(f"警告: 窗口 {window_idx} 資料不足，跳過 "
                  f"(訓練: {len(window_data['train_input'])}，"
                  f"驗證: {len(window_data['valid_input'])}，"
                  f"測試: {len(window_data['test_input'])})")
            # 如果資料不足，不要增加窗口索引，重試
            continue
        
        # 進入下一個窗口
        window_idx += 1
    
    # 保存所有窗口的資料
    save_dir = "/Users/chuchu/Desktop/option_pricing_thesis_project/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(all_windows, f"{save_dir}/rolling_windows_data.pt")
    
    print(f"成功創建了 {len(all_windows)} 個累積式滾動窗口的資料")
    return all_windows

if __name__ == "__main__":
    config = {"prepare splited data": True, "prepare rolling data":False}
    data = data_preprocess()
    if config.get("prepare for rolling data"):
        prepare_rolling_window_data(data)
    elif config.get("prepare splited data"):
        prepare_data(data)
