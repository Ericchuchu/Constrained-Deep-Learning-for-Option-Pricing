import argparse
import numpy as np
import torch
from os.path import isdir
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from core.models_convlstm import ConvLSTM
from core.functions import *
from torch.utils import data
from torch import nn, optim
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_data(data_input, data_label, timestamp_tensor, args):
    # data_input:(N, C, T, D), data_label:(N, T, 1)
    data_loader = data.DataLoader(data.TensorDataset(data_input, data_label, timestamp_tensor), batch_size=args.batch_size, shuffle=False, drop_last=False)
    return data_loader

def get_models(args):
    if args.test_checkpoint is None:
        net = ConvLSTM(input_channels=3, hidden_channels=[16, 8, 1], kernel_size=5, in_dim=5, out_dim=1, step=10)
        optimizer = optim.Adam(list(net.parameters()), amsgrad=True, lr=args.learning_rate)
    else:
        try:
            net = torch.load("checkpoints/" + args.test_checkpoint + "_net")
            optimizer = torch.load("checkpoints/" + args.test_checkpoint + "_optimizer")
        except FileNotFoundError:
            try:
                net = torch.load(args.test_checkpoint + "_net")
                optimizer = torch.load(args.test_checkpoint + "_optimizer")
            except:
                print("No checkpoint found at '{}'- Please specify the model for testing".format(args.test_checkpoint))
                exit()

    if torch.cuda.is_available():
        net.cuda()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(device)

    return net, optimizer

def evaluate(model, data_loader, criterion, dict, last_epoch = False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, timestamp in data_loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss_value = loss.detach().item()
            total_loss += loss.detach().item()

            if last_epoch:
                for ts in timestamp:
                    date_str = f"{int(ts[0]):04d}{int(ts[1]):02d}{int(ts[2]):02d}"
                    if date_str in dict:
                        dict[date_str].append(loss_value)
                    else:
                        dict[date_str] = [loss_value]
    
    return total_loss / len(data_loader),dict

def train(train_loader, valid_loader, test_loader, args):
    net, optimizer = get_models(args)
    criterion = nn.MSELoss()

    # make a directory to save models if it doesn't exist
    if not isdir("checkpoints"):
        os.mkdir("checkpoints")

    print("Training the model")
    train_loss_dict = {} # 儲存 timestamp 以及對應的 loss 
    valid_loss_dict = {}
    test_loss_dict = {}
    last_epoch = False
    loss_var = []
    valid_losses = []
    test_losses = []
    trigger_times = 0
    patience = 5
    early_stop = False
    the_last_loss = float('inf')
    the_last_test_loss = float('inf')
    minimum_test_loss_epoch = 0

    for epoch in tqdm(range(args.max_epoch)):
        net.train()
        train_loss = []
        for x, y, timestamp in train_loader:
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss_value = loss.detach().item()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().clone())

            # 迭帶當前 batch 中的 timestamp
            if epoch == (args.max_epoch - 1):
                for ts in timestamp:
                    date_str = f"{int(ts[0]):04d}{int(ts[1]):02d}{int(ts[2]):02d}"
                    if date_str in train_loss_dict:
                        train_loss_dict[date_str].append(loss_value)
                    else:
                        train_loss_dict[date_str] = [loss_value]

        last_epoch = epoch == (args.max_epoch - 1)
        # Evaluate on validation set
        valid_loss, valid_loss_dict = evaluate(net, valid_loader, criterion, valid_loss_dict, last_epoch)
        valid_losses.append(valid_loss)

        # Evaluate on test set
        test_loss, test_loss_dict = evaluate(net, test_loader, criterion, test_loss_dict, last_epoch)
        test_losses.append(test_loss)

        train_loss = torch.tensor(train_loss)
        epoch_loss = torch.mean(train_loss).squeeze()
        loss_var.append(epoch_loss)

        # 計算 timestamp 中的平均 loss 值
        for key in train_loss_dict:
            train_loss_dict[key] = sum(train_loss_dict[key]) / len(train_loss_dict[key])
        for key in valid_loss_dict:
            valid_loss_dict[key] = sum(valid_loss_dict[key]) / len(valid_loss_dict[key])
        for key in test_loss_dict:
            test_loss_dict[key] = sum(test_loss_dict[key]) / len(test_loss_dict[key])

        # minimum test loss
        if test_loss < the_last_test_loss:
            minimum_test_loss_epoch = epoch 
            the_last_test_loss = test_loss
        
        # Early stopping
        if args.early_stop_mode:
            if valid_loss > the_last_loss:
                trigger_times += 1
                # print('trigger times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping')
                    torch.save(net, "checkpoints/{}_net".format(args.session_name))
                    torch.save(optimizer, "checkpoints/{}_optimizer".format(args.session_name))
                    early_stop = True
                    early_stop_epoch = epoch
                    break

            else:
                # print('trigger times: 0')
                trigger_times = 0

            the_last_loss = valid_loss

    # 沒有early stop, epoch跑完時
    if not early_stop:
        torch.save(net, "checkpoints/{}_net".format(args.session_name))
        torch.save(optimizer, "checkpoints/{}_optimizer".format(args.session_name))
        epochs = np.arange(args.max_epoch)
    else:
        epochs = np.arange(early_stop_epoch+1)

    loss_var = torch.tensor(loss_var)
    loss_var = loss_var.numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_var, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    if early_stop:
        plt.axvline(x=early_stop_epoch, color='r', linestyle='--', label='Early Stopping')
        plt.text(early_stop_epoch, plt.ylim()[1], 'Early Stop', color = 'red', horizontalalignment='right')
    plt.axvline(x=minimum_test_loss_epoch, color='g', linestyle='--', label='Minimum Test Loss')
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # 保存圖片
    plt.savefig(f'loss_curves/loss_curves_convlstm_epoch{args.max_epoch}_early_stop{args.early_stop_mode}_syn{args.syn_data}.png')
    plt.show()

    return train_loss_dict, valid_loss_dict, test_loss_dict

def test(test_loader, args):
    if args.test_checkpoint is None:
        args.test_checkpoint = "checkpoints/{}".format(args.session_name)
    model, _ = get_models(args)
    criterion = nn.MSELoss()
    model.eval()

    true_price = []
    estimate_price = []

    test_loss = []
    test_map = []
    test_mape = []
    test_property_corr = []
    strike_price = []
    time_to_maturity = []

    model.eval()
    print("\nTesting the model")
    with torch.no_grad():
        for x, y, timestamp in test_loader:
            y_hat = model(x)
            y = test_label_normalization.unnormalize(y)
            y_hat = test_label_normalization.unnormalize(y_hat)
            loss = criterion(y, y_hat)
            corr, map, mape = metrics(y_hat.detach(), y.detach())

            test_loss.append(loss.item())
            test_map.append(map)
            test_mape.append(mape)
            test_property_corr.append(corr)

            true_price.append(y.detach())
            estimate_price.append(y_hat.detach())
            x = test_input_normalization.unnormalize(x)
            strike_price.append(x[:,0,-1,4].detach())
            time_to_maturity.append(x[:,0,-1,2].detach())

        property_corr = torch.mean(torch.cat(test_property_corr, dim=0)).squeeze().float()
        map = torch.mean(torch.cat(test_map, dim=0)).squeeze().float()
        mape = torch.mean(torch.cat(test_mape, dim=0)).squeeze().float()
        loss = torch.mean(torch.tensor(test_loss)).float()
        rmse = torch.sqrt(loss)
        print("MSE: {:.5f}\nRMSE: {:0.5f}\nMAP: {:0.5f}\nMAPE: {:0.5f}\nCorrelation: {:0.5f}\n".format(loss, rmse, map, mape, property_corr))

        # shape:(N, T, D_out=3)
        true_price = torch.cat(true_price, dim=0)
        estimate_price = torch.cat(estimate_price, dim=0)
        strike_price = torch.cat(strike_price, dim = 0)
        time_to_maturity = torch.cat(time_to_maturity, dim = 0)

        true_price = true_price.cpu()
        estimate_price = estimate_price.cpu()
        strike_price = strike_price.cpu()
        time_to_maturity = time_to_maturity.cpu()

        true_price = true_price.numpy()
        estimate_price = estimate_price.numpy()
        strike_price = strike_price.numpy()
        time_to_maturity = time_to_maturity.numpy()

        # 三維圖表
        plot_len = 120
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(strike_price[0:plot_len], time_to_maturity[0:plot_len], true_price[0:plot_len], color='b', label='True Price')
        ax.scatter(strike_price[0:plot_len], time_to_maturity[0:plot_len], estimate_price[0:plot_len], color='r', label='Estimated Price')
        # 標題和座標軸
        ax.set_title('True vs. Estimated Option Prices')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Option Price')
        ax.legend()
        # 保存圖片
        plt.savefig(f'testing_result/testing_result_convlstm_epoch{args.max_epoch}_early_stop{args.early_stop_mode}_syn{args.syn_data}.png')
        plt.show()

def plot_timestamp_loss(train_loss_dict, valid_loss_dict, test_loss_dict, args):

    # 提取所有的日期
    all_dates = sorted(list(set(train_loss_dict.keys()) | set(valid_loss_dict.keys()) | set(test_loss_dict.keys())))

    # 創建一個字典來存儲每個日期對應的損失值
    train_loss_values = {date: train_loss_dict.get(date, None) for date in all_dates}
    valid_loss_values = {date: valid_loss_dict.get(date, None) for date in all_dates}
    test_loss_values = {date: test_loss_dict.get(date, None) for date in all_dates}

    # 創建圖表
    fig, ax = plt.subplots(figsize=(14, 7))

    # 繪製訓練損失
    ax.plot(all_dates, [train_loss_values[date] for date in all_dates], marker='o', linestyle='-', color='blue', label='Train Loss')

    # 繪製驗證損失
    ax.plot(all_dates, [valid_loss_values[date] for date in all_dates], marker='o', linestyle='-', color='green', label='Validation Loss')

    # 繪製測試損失
    ax.plot(all_dates, [test_loss_values[date] for date in all_dates], marker='o', linestyle='-', color='red', label='Test Loss')


    ax.set_title('Training, Validation, and Test Loss Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Loss Value')
    ax.set_xticks(all_dates[::10])  
    ax.set_xticklabels(all_dates[::10], rotation=45)  

    ax.legend()

    plt.tight_layout()

    plt.savefig(f'loss_curves/timestamp_loss_curves_convlstm_epoch{args.max_epoch}_early_stop{args.early_stop_mode}_syn{args.syn_data}.png')

def plot_timestamp_loss_modified(train_loss_dict, valid_loss_dict, test_loss_dict, args):
    # 提取test_dataset的日期
    test_dates = sorted(test_loss_dict.keys())

    # 排除日期為 20200507 和 20200514 的數據
    excluded_dates = ['20210507', '20210514']
    test_dates = [date for date in test_dates if date not in excluded_dates]
    
    # 創建字典來存儲每個測試日期對應的損失值
    train_loss_values = {}
    valid_loss_values = {}
    test_loss_values = {}
    
    for test_date in test_dates:
        # valid
        valid_dates = [date for date in valid_loss_dict.keys() if date < test_date][-args.valid_days:]
        valid_losses = [valid_loss_dict[date] for date in valid_dates]
        valid_loss_values[test_date] = np.mean(valid_losses) 
        # train
        train_dates = [date for date in train_loss_dict.keys() if date < test_date][-args.train_days:]
        train_losses = [train_loss_dict[date] for date in train_dates]
        train_loss_values[test_date] = np.mean(train_losses) 
        # test
        test_loss_values[test_date] = test_loss_dict[test_date]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(test_dates, [train_loss_values[date] for date in test_dates], marker='o', linestyle='-', color='blue', label='Avg Train Loss (3 days)')
    ax.plot(test_dates, [valid_loss_values[date] for date in test_dates], marker='o', linestyle='-', color='green', label='Validation Loss (1 day before)')
    ax.plot(test_dates, [test_loss_values[date] for date in test_dates], marker='o', linestyle='-', color='red', label='Test Loss')
    

    ax.set_title('Training, Validation, and Test Loss Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Loss Value')
    ax.set_xticks(test_dates)  
    ax.set_xticklabels(test_dates, rotation=45)  

    ax.legend()

    plt.tight_layout()

    plt.savefig(f'loss_curves/timestamp_loss_curves_convlstm_epoch{args.max_epoch}_early_stop{args.early_stop_mode}_syn{args.syn_data}.png')

if __name__ == '__main__':
    ## Arguments and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-learning_rate', type=float, default=0.0001, help="learning rate of the Adam")
    parser.add_argument('-max_epoch', type=int, default=100, help="maximum number of training epochs")
    parser.add_argument('-batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'),
                        help="name of the session to be used in saving the model")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None,
                        help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-nonlinearity', action="store", type=str, default="tanh",
                        help="Type of nonlinearity for the CNN [tanh, relu]", choices=["tanh", "relu"])
    parser.add_argument('-early_stop_mode',type=bool, default=False, help="training the model with early stop mode" )
    parser.add_argument('-train_days', type=int, default=3, help="the days for training")
    parser.add_argument('-valid_days', type=int, default=1, help="the days for validating")
    parser.add_argument('-test_days', type=int, default=1, help="the days for testing")
    parser.add_argument('-syn_data', type=bool, default=True, help="synthesize the missing data from TAIEX website")

    args = parser.parse_args()

    if args.syn_data:
        train_input = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_input_taiex2_syn.pt", weights_only=True).float()
    else:
        train_input = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_input_taiex2.pt", weights_only=True).float()
    train_input_mean = torch.mean(train_input, dim=0, keepdim=True).float()
    train_input_std = torch.std(train_input, dim=0, keepdim=True).float()
    train_input_std += torch.tensor(1e-9)
    train_input_normalization = Normalization(train_input_mean, train_input_std)
    train_input = train_input_normalization.normalize(train_input)

    if args.syn_data:
        train_label = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_label_taiex2_syn.pt", weights_only=True).float()
        train_label_timestamp = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_label_timestamp_taiex2_syn.pt", weights_only=True).float()
    else:
        train_label = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_label_taiex2.pt", weights_only=True).float()
        train_label_timestamp = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_label_timestamp_taiex2.pt", weights_only=True).float()
    train_label_mean = torch.mean(train_label, dim=0, keepdim=False)
    train_label_std = torch.std(train_label, dim=0, keepdim=False) 
    train_label_normalization = Normalization(train_label_mean, train_label_std)
    train_label = train_label_normalization.normalize(train_label)

    if args.syn_data:
        test_input = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_input_taiex2_syn.pt", weights_only=True).float()
    else:
        test_input = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_input_taiex2.pt", weights_only=True).float()
    test_input_mean = torch.mean(test_input, dim=0, keepdim=True)
    test_input_std = torch.std(test_input, dim=0, keepdim=True)
    test_input_std += torch.tensor(1e-9)
    test_input_normalization = Normalization(test_input_mean, test_input_std)
    test_input = test_input_normalization.normalize(test_input)

    if args.syn_data:
        test_label = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_label_taiex2_syn.pt", weights_only=True).float()
        test_label_timestamp = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_label_timestamp_taiex2_syn.pt", weights_only=True).float()
    else:
        test_label = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_label_taiex2.pt", weights_only=True).float()
        test_label_timestamp = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_label_timestamp_taiex2.pt", weights_only=True).float()
    test_label_mean = torch.mean(test_label, dim=0, keepdim=False)
    test_label_std = torch.std(test_label, dim=0, keepdim=False)
    test_label_normalization = Normalization(test_label_mean, test_label_std)
    test_label = test_label_normalization.normalize(test_label)

    if args.syn_data:
        valid_input = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_input_taiex2_syn.pt", weights_only=True).float()
    else:
        valid_input = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_input_taiex2.pt", weights_only=True).float()
    valid_input_mean = torch.mean(valid_input, dim=0, keepdim=True)
    valid_input_std = torch.std(valid_input, dim=0, keepdim=True)
    valid_input_std += torch.tensor(1e-9)
    valid_input_normalization = Normalization(valid_input_mean, valid_input_std)
    valid_input = valid_input_normalization.normalize(valid_input)

    if args.syn_data:
        valid_label = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_label_taiex2_syn.pt", weights_only=True).float()
        valid_label_timestamp = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_label_timestamp_taiex2_syn.pt", weights_only=True).float()
    else:
        valid_label = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_label_taiex2.pt", weights_only=True).float()
        valid_label_timestamp = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_label_timestamp_taiex2.pt", weights_only=True).float()
    valid_label_mean = torch.mean(valid_label, dim=0, keepdim=False)
    valid_label_std = torch.std(valid_label, dim=0, keepdim=False)
    valid_label_normalization = Normalization(valid_label_mean, valid_label_std)
    valid_label = valid_label_normalization.normalize(valid_label)
  

    train_loader = get_data(train_input, train_label, train_label_timestamp, args)
    test_loader = get_data(test_input, test_label, test_label_timestamp, args)
    valid_loader = get_data(valid_input, valid_label, valid_label_timestamp, args)

    train_loss_dict, valid_loss_dict, test_loss_dict = train(train_loader, valid_loader, test_loader, args)
    test(test_loader, args)

    plot_timestamp_loss_modified(train_loss_dict, valid_loss_dict, test_loss_dict, args)


    # use the following code if test:
    # parser.set_defaults(test_checkpoint="May16_174652")
    # args = parser.parse_args()
    # test(test_loader, args)




















