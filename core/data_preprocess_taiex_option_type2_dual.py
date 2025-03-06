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
    output_file_path = r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/prs_dataset_dual.csv"

    # Check if preprocessed file exists
    if os.path.exists(output_file_path):
        data = pd.read_csv(output_file_path)
        data['date'] = pd.to_datetime(data['date'])
        data['exdate'] = pd.to_datetime(data['exdate'])
        print("Dataset loaded from prs_dataset_dual.csv")
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

        # Transform moneyness based on option type
        data['transformed_moneyness'] = data.apply(
            lambda row: -log(row['invm'])/(row['impl_volatility']*sqrt(row['tau'])) if row['PC'] == 'C' 
            else log(row['invm'])/(row['impl_volatility']*sqrt(row['tau'])), 
            axis=1
        )

        data['opt_ID'] = data.groupby(['strike_price', 'exdate', 'PC']).ngroup()
        data['pre_settle_price'] = data.groupby('opt_ID')['option_price'].shift(1)
        data['settle_price_chg'] = ((data['option_price'] - data['pre_settle_price']) / data['pre_settle_price']) * 100
        data['theory_margin'] = data['theory_price'] - data['option_price']
        
        # Convert PC to numeric
        data['PC'] = data['PC'].map({'C': 0, 'P': 1})
        
        data = data.dropna().reset_index(drop=True)
        data.to_csv(output_file_path)
        print("Dataset preprocessed and saved to prs_dataset_dual.csv")
    
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
    input_convlstm = torch.zeros((3, seq_len, 5))
    input_transformer = torch.zeros((seq_len, 2))  # [transformed_moneyness, date_position]
    
    tmp_input = data.iloc[index:index+seq_len+1]
    if len(tmp_input) < seq_len + 1:
        return None, None, None, None
        
    # ConvLSTM input
    input_convlstm[0, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['volume','PC','tau','impl_volatility','strike_price']], dtype=np.float64))
    input_convlstm[1, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['delta','gamma','rho','theta','vega']], dtype=np.float64))
    input_convlstm[2, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['pre_settle_price','settle_price_chg','S','theory_margin','theory_price']], dtype=np.float64))
    
    # Transformer input - transformed moneyness and date position
    input_transformer[:, 0] = torch.tensor(np.array(tmp_input.iloc[0:seq_len]['transformed_moneyness'], dtype=np.float64))
    input_transformer[:, 1] = torch.tensor(np.array(tmp_input.iloc[0:seq_len]['date_position'], dtype=np.float64))
    
    label = torch.tensor(np.array(tmp_input.iloc[seq_len]['option_price'], dtype=np.float64))
    date_tensor = torch.tensor(np.array([tmp_input.iloc[seq_len]['date'].year, 
                                       tmp_input.iloc[seq_len]['date'].month, 
                                       tmp_input.iloc[seq_len]['date'].day], dtype=np.int64))
    
    return input_convlstm, input_transformer, label, date_tensor

def prepare_data(data, seq_len=10, train_days=3, valid_days=1, test_days=1):
    train_input_convlstm = []
    train_input_transformer = []
    train_label = []
    valid_input_convlstm = []
    valid_input_transformer = []
    valid_label = []
    test_input_convlstm = []
    test_input_transformer = []
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
                    input_convlstm, input_transformer, label, date_tensor = build_input_label(train_data, j, seq_len)
                    if input_convlstm is not None:
                        train_input_convlstm.append(input_convlstm)
                        train_input_transformer.append(input_transformer)
                        train_label.append(label)
                        train_label_timestamp.append(date_tensor)
            
            # Process validation data
            if len(valid_data) >= seq_len + 1:
                for j in range(len(valid_data) - seq_len):
                    input_convlstm, input_transformer, label, date_tensor = build_input_label(valid_data, j, seq_len)
                    if input_convlstm is not None:
                        valid_input_convlstm.append(input_convlstm)
                        valid_input_transformer.append(input_transformer)
                        valid_label.append(label)
                        valid_label_timestamp.append(date_tensor)
            
            # Process test data
            if len(test_data) >= seq_len + 1:
                for j in range(len(test_data) - seq_len):
                    input_convlstm, input_transformer, label, date_tensor = build_input_label(test_data, j, seq_len)
                    if input_convlstm is not None:
                        test_input_convlstm.append(input_convlstm)
                        test_input_transformer.append(input_transformer)
                        test_label.append(label)
                        test_label_timestamp.append(date_tensor)

    # Convert lists to tensors
    train_input_convlstm = torch.stack(train_input_convlstm)
    train_input_transformer = torch.stack(train_input_transformer)
    train_label = torch.stack(train_label)
    train_label_timestamp = torch.stack(train_label_timestamp)

    valid_input_convlstm = torch.stack(valid_input_convlstm)
    valid_input_transformer = torch.stack(valid_input_transformer)
    valid_label = torch.stack(valid_label)
    valid_label_timestamp = torch.stack(valid_label_timestamp)

    test_input_convlstm = torch.stack(test_input_convlstm)
    test_input_transformer = torch.stack(test_input_transformer)
    test_label = torch.stack(test_label)
    test_label_timestamp = torch.stack(test_label_timestamp)

    # Save tensors
    save_dir = "/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data"
    torch.save(train_input_convlstm, f"{save_dir}/train_input_convlstm_dual.pt")
    torch.save(train_input_transformer, f"{save_dir}/train_input_transformer_dual.pt")
    torch.save(train_label, f"{save_dir}/train_label_dual.pt")
    torch.save(train_label_timestamp, f"{save_dir}/train_label_timestamp_dual.pt")
    
    torch.save(valid_input_convlstm, f"{save_dir}/valid_input_convlstm_dual.pt")
    torch.save(valid_input_transformer, f"{save_dir}/valid_input_transformer_dual.pt")
    torch.save(valid_label, f"{save_dir}/valid_label_dual.pt")
    torch.save(valid_label_timestamp, f"{save_dir}/valid_label_timestamp_dual.pt")
    
    torch.save(test_input_convlstm, f"{save_dir}/test_input_convlstm_dual.pt")
    torch.save(test_input_transformer, f"{save_dir}/test_input_transformer_dual.pt")
    torch.save(test_label, f"{save_dir}/test_label_dual.pt")
    torch.save(test_label_timestamp, f"{save_dir}/test_label_timestamp_dual.pt")

    return (train_input_convlstm, train_input_transformer, train_label, train_label_timestamp,
            valid_input_convlstm, valid_input_transformer, valid_label, valid_label_timestamp,
            test_input_convlstm, test_input_transformer, test_label, test_label_timestamp)

if __name__ == "__main__":
    data = data_preprocess()
    prepare_data(data)
