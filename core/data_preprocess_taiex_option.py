import pandas as pd
import numpy as np
import torch
from tqdm import trange
from functions import *
import os

# 文件路径
input_file_path = r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/TXO_with_TXF.csv"
output_file_path = r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/prs_dataset.csv"

# 检查文件是否存在
if os.path.exists(output_file_path):
    # 如果文件存在，直接读取
    data = pd.read_csv(output_file_path)
    print("Dataset loaded from prs_dataset.csv")

else:
    data = pd.read_csv(input_file_path)
    data['r'] = data['r']/100
    # 去除 PC 列中的前後空格
    data['PC'] = data['PC'].str.strip()

    # 計算相關值
    data['impl_volatility'] = data.apply(calculate_implied_volatility, axis = 1)
    data['delta'] = data.apply(BSM_delta, axis = 1)
    data['gamma'] = data.apply(BSM_gamma, axis = 1)
    data['theta'] = data.apply(BSM_theta, axis = 1)
    data['rho'] = data.apply(BSM_rho, axis = 1)
    data['vega'] = data.apply(BSM_vega, axis = 1)
    data['theory_price'] = data.apply(calculate_theory_price, axis = 1)

    data['date'] = pd.to_datetime(data['date'])
    data['opt_ID'] = data.groupby(['strike_price', 'exdate']).ngroup()
    data = data.sort_values(by=['opt_ID', 'date']).reset_index(drop=True)
    data['pre_settle_price'] = data.groupby('opt_ID')['option_price'].shift(1)
    data['settle_price_chg'] = ((data['option_price'] - data['pre_settle_price']) / data['pre_settle_price']) * 100
    data['theory_margin'] = data['theory_price'] - data['option_price']
    data['PC'] = data['PC'].map({'C': 0, 'P': 1})
    data = data.dropna().reset_index(drop=True)
    data.to_csv(output_file_path)
    print("Dataset preprocessed and saved to prs_dataset.csv")

data = data[data['date'] >= '2021-01-01']
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]
train_input = []
train_label = []
seq_len = 10

for i in trange(train_data['opt_ID'].unique().shape[0]):
    if (train_data.loc[(train_data['opt_ID'] == train_data['opt_ID'].unique()[i])].shape[0] - seq_len) >= 0:
        for index in range(train_data.loc[(train_data['opt_ID'] == train_data['opt_ID'].unique()[i])].shape[0] - seq_len):
            input = torch.zeros((3, 10, 5))
            # 分為三個 channel 
            tmp_input = train_data.loc[(train_data['opt_ID'] == train_data['opt_ID'].unique()[i])][index:index+seq_len+1]
            input[0, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['volume','PC','tau','impl_volatility','strike_price']], dtype=np.float64))
            # 基本面 inventory   call(label 0 )/put(label 1 )   tau(days)   impli_vol   strikePrice
            input[1, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['delta','gamma','rho','theta','vega']], dtype=np.float64))
            # 希臘字母 delta   gamma   rho   theta   vega
            input[2, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['pre_settle_price','settle_price_chg','S','theory_margin','theory_price']], dtype=np.float64))
            # 價格 pre_settlePrice   settlePrice_chg   spotPrice   theoryMargin   theoryPrice
            label = torch.tensor(np.array(tmp_input.iloc[seq_len]['option_price'], dtype=np.float64))
            train_input.append(input)
            train_label.append(label)

# 將 train_input 轉為 torch tensor
zeros = torch.zeros((len(train_input), 3, 10, 5))
for i in range(zeros.shape[0]):
    zeros[i] = train_input[i]
train_input = zeros
print("Shape of the Training Input:", train_input.shape)

train_label = torch.tensor(train_label)
print("Shape of the Training Label:", train_label.shape)

test_valid_input = []
test_valid_label = []

for i in trange(test_data['opt_ID'].unique().shape[0]):
    if (test_data.loc[(test_data['opt_ID'] == test_data['opt_ID'].unique()[i])].shape[0] - seq_len) >= 0:
        for index in range(test_data.loc[(test_data['opt_ID'] == test_data['opt_ID'].unique()[i])].shape[0] - seq_len):
            input = torch.zeros((3, 10, 5))
            tmp_input = test_data.loc[(test_data['opt_ID'] == test_data['opt_ID'].unique()[i])][index:index+seq_len+1]
            input[0, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['volume','PC','tau','impl_volatility','strike_price']], dtype=np.float64))
            # 基本面 inventory   call(label 0 )/put(label 1 )   tau(days)   impli_vol   strikePrice
            input[1, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['delta','gamma','rho','theta','vega']], dtype=np.float64))
            # 希臘字母 delta   gamma   rho   theta   vega
            input[2, :] = torch.tensor(np.array(tmp_input.iloc[0:seq_len][['pre_settle_price','settle_price_chg','S','theory_margin','theory_price']], dtype=np.float64))
            # 價格 pre_settlePrice   settlePrice_chg   spotPrice   theoryMargin   theoryPrice
            label = torch.tensor(np.array(tmp_input.iloc[seq_len]['option_price'], dtype=np.float64))
            test_valid_input.append(input)
            test_valid_label.append(label)

valid_input = test_valid_input[int(0.5*len(test_valid_input)):]
valid_label = test_valid_label[int(0.5*len(test_valid_input)):]
test_input = test_valid_input[:int(0.5*len(test_valid_input))]
test_label = test_valid_label[:int(0.5*len(test_valid_input))]

zeros = torch.zeros((len(valid_input), 3, 10, 5))
for i in range(zeros.shape[0]):
    zeros[i] = valid_input[i]
valid_input = zeros
print("Shape of the Valid Input:", valid_input.shape)

valid_label = torch.tensor(valid_label)
print("Shape of the Valid Label:", valid_label.shape)

zeros = torch.zeros((len(test_input), 3, 10, 5))
for i in range(zeros.shape[0]):
    zeros[i] = test_input[i]
test_input = zeros
print("Shape of the Test Input:", test_input.shape)

test_label = torch.tensor(test_label)
print("Shape of the Test Label:", test_label.shape)
torch.save(train_input, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_input_taiex.pt")
torch.save(train_label, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_label_taiex.pt")
torch.save(test_input, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_input_taiex.pt")
torch.save(test_label, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_label_taiex.pt")
torch.save(valid_input, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_input_taiex.pt")
torch.save(valid_label, r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_label_taiex.pt")

