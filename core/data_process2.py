import pandas as pd
import numpy as np
import torch
from tqdm import trange

train_data = pd.read_csv(r"C:\Users\Admin\Desktop\option pricing專題\conv-LSTM run code\3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main\data\train_data.csv")
test_data = pd.read_csv(r"C:\Users\Admin\Desktop\option pricing專題\conv-LSTM run code\3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main\data\test_data.csv")

train_input = []
train_label = []
seq_len = 10

for i in trange(train_data['optID'].unique().shape[0]):
    for index in range(train_data.loc[(train_data['optID'] == train_data['optID'].unique()[i])].shape[0] - seq_len):
        input = torch.zeros((1, 10, 15))
        # 分為三個 channel 
        tmp_input = np.array(train_data.loc[(train_data['optID'] == train_data['optID'].unique()[i])][index:index+seq_len+1])
        input[0, :] = torch.tensor(np.array(tmp_input[0:seq_len, 3:18], dtype=np.float64))
        label = torch.tensor(np.array(tmp_input[seq_len, 18], dtype=np.float64))
        train_input.append(input)
        train_label.append(label)

# 將 train_input 轉為 torch tensor
zeros = torch.zeros((len(train_input), 1, 10, 15))
for i in range(zeros.shape[0]):
    zeros[i] = train_input[i]
train_input = zeros
print("Shape of the Training Input:", train_input.shape)

train_label = torch.tensor(train_label)
print("Shape of the Training Label:", train_label.shape)

test_valid_input = []
test_valid_label = []


for i in trange(test_data['optID'].unique().shape[0]):
    for index in range(test_data.loc[(test_data['optID'] == test_data['optID'].unique()[i])].shape[0] - seq_len):
        input = torch.zeros((1, 10, 15))
        tmp_input = np.array(test_data.loc[(test_data['optID'] == test_data['optID'].unique()[i])][index:index+seq_len+1])
        input[0, :] = torch.tensor(np.array(tmp_input[0:seq_len, 3:18], dtype=np.float64))
        label = torch.tensor(np.array(tmp_input[seq_len, 18], dtype=np.float64))
        test_valid_input.append(input)
        test_valid_label.append(label)

valid_input = test_valid_input[int(0.5*len(test_valid_input)):]
valid_label = test_valid_label[int(0.5*len(test_valid_input)):]
test_input = test_valid_input[:int(0.5*len(test_valid_input))]
test_label = test_valid_label[:int(0.5*len(test_valid_input))]

zeros = torch.zeros((len(valid_input), 1, 10, 15))
for i in range(zeros.shape[0]):
    zeros[i] = valid_input[i]
valid_input = zeros
print("Shape of the Valid Input:", valid_input.shape)

valid_label = torch.tensor(valid_label)
print("Shape of the Valid Label:", valid_label.shape)

zeros = torch.zeros((len(test_input), 1, 10, 15))
for i in range(zeros.shape[0]):
    zeros[i] = test_input[i]
test_input = zeros
print("Shape of the Test Input:", test_input.shape)

test_label = torch.tensor(test_label)
print("Shape of the Test Label:", test_label.shape)
torch.save(train_input, r"C:\Users\Admin\Desktop\option pricing專題\conv-LSTM run code\3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main\data\torch-data\train_input_convlstm.pt")
torch.save(train_label, r"C:\Users\Admin\Desktop\option pricing專題\conv-LSTM run code\3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main\data\torch-data\train_label_convlstm.pt")
torch.save(test_input, r"C:\Users\Admin\Desktop\option pricing專題\conv-LSTM run code\3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main\data\torch-data\test_input_convlstm.pt")
torch.save(test_label, r"C:\Users\Admin\Desktop\option pricing專題\conv-LSTM run code\3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main\data\torch-data\test_label_convlstm.pt")
torch.save(valid_input, r"C:\Users\Admin\Desktop\option pricing專題\conv-LSTM run code\3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main\data\torch-data\valid_input_convlstm.pt")
torch.save(valid_label, r"C:\Users\Admin\Desktop\option pricing專題\conv-LSTM run code\3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main\data\torch-data\valid_label_convlstm.pt")
