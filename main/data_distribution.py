import torch
import matplotlib.pyplot as plt

train_label_timestamp = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_label_timestamp_taiex2_syn.pt").float()
test_label_timestamp = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_label_timestamp_taiex2_syn.pt").float()
valid_label_timestamp = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_label_timestamp_taiex2_syn.pt").float()
train_data = []
valid_data = []
test_data = []
for ts in train_label_timestamp:
    date_str = f"{int(ts[0]):04d}{int(ts[1]):02d}{int(ts[2]):02d}"
    train_data.append(date_str)
for ts in valid_label_timestamp:
    date_str = f"{int(ts[0]):04d}{int(ts[1]):02d}{int(ts[2]):02d}"
    valid_data.append(date_str)
for ts in test_label_timestamp:
    date_str = f"{int(ts[0]):04d}{int(ts[1]):02d}{int(ts[2]):02d}"
    test_data.append(date_str)

timestamps = sorted(list(set(train_data + test_data + valid_data)))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(40, 30))

# 在第一個子圖中繪製 train 數據的數據筆數
train_counts = [train_data.count(ts) for ts in timestamps]
ax1.bar(timestamps, train_counts)
ax1.set_title('Train Data')
ax1.set_xlabel('Date')
ax1.set_ylabel('Data Count')
ax1.tick_params(axis='x', rotation=90)

# 在第二個子圖中繪製 valid 數據的數據筆數
valid_counts = [valid_data.count(ts) for ts in timestamps]
ax2.bar(timestamps, valid_counts)
ax2.set_title('Valid Data')
ax2.set_xlabel('Date')
ax2.set_ylabel('Data Count')
ax2.tick_params(axis='x', rotation=90)

# 在第三個子圖中繪製 test 數據的數據筆數
test_counts = [test_data.count(ts) for ts in timestamps]
ax3.bar(timestamps, test_counts)
ax3.set_title('Test Data')
ax3.set_xlabel('Date')
ax3.set_ylabel('Data Count')
ax3.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

plt.savefig(f'data_distribution/data_distribution.png')