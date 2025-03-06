import argparse
import numpy as np
import torch
from os.path import isdir
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from core.models_lstm import LSTM
from core.functions import *
from torch.utils import data
from torch import nn, optim
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_data(data_input, data_label, args):
    # data_input:(N, C, T, D), data_label:(N, T, 1)
    data_loader = data.DataLoader(data.TensorDataset(data_input, data_label), batch_size=args.batch_size, shuffle=False, drop_last=False)
    return data_loader

def get_models(args):
    if args.test_checkpoint is None:
        net = LSTM(in_dim=15, in_channels=1, out_dim=1, seq_len=10)
        optimizer = optim.Adam(list(net.parameters()), amsgrad=True, lr=args.learning_rate)
    else:
        try:
            net = torch.load("/main/checkpoints/" + args.test_checkpoint + "_net")
            optimizer = torch.load("/main/checkpoints/" + args.test_checkpoint + "_optimizer")
        except FileNotFoundError:
            try:
                net = torch.load(args.test_checkpoint + "_net")
                optimizer = torch.load(args.test_checkpoint + "_optimizer")
            except:
                print("No checkpoint found at '{}'- Please specify the model for testing".format(args.test_checkpoint))
                exit()

    if torch.cuda.is_available():
        net.cuda()

    return net, optimizer

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.detach().item()
    return total_loss / len(data_loader)

def train(train_loader, valid_loader, test_loader, args):
    net, optimizer = get_models(args)
    net.train()
    criterion = nn.MSELoss()

    # make a directory to save models if it doesn't exist
    if not isdir("checkpoints"):
        os.mkdir("checkpoints")

    print("Training the model")
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
        train_loss = []
        for x, y in train_loader:
            net.train()
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().clone())

        # Evaluate on validation set
        valid_loss = evaluate(net, valid_loader, criterion)
        valid_losses.append(valid_loss)

        # Evaluate on test set
        test_loss = evaluate(net, test_loader, criterion)
        test_losses.append(test_loss)

        train_loss = torch.tensor(train_loss)
        epoch_loss = torch.mean(train_loss).squeeze()
        loss_var.append(epoch_loss)
        
        # minimum test loss
        if test_loss < the_last_test_loss:
            minimum_test_loss_epoch = epoch 
            the_last_test_loss = test_loss
        
        # Early stopping
        if args.early_stop_mode:
            if valid_loss  > the_last_loss:
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
    plt.savefig(f'loss_curves/loss_curves_lstm_epoch{args.max_epoch}_early_stop{args.early_stop_mode}.png')
    plt.show()


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
        for x, y in test_loader:
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
        plt.savefig(f'testing_result/testing_result_lstm_epoch{args.max_epoch}_early_stop{args.early_stop_mode}.png')
        plt.show()

if __name__ == '__main__':
    ## Arguments and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-learning_rate', type=float, default=0.0001, help="learning rate of the Adam")
    parser.add_argument('-max_epoch', type=int, default=400, help="maximum number of training epochs")
    parser.add_argument('-batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'),
                        help="name of the session to be used in saving the model")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None,
                        help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-nonlinearity', action="store", type=str, default="tanh",
                        help="Type of nonlinearity for the CNN [tanh, relu]", choices=["tanh", "relu"])
    parser.add_argument('-early_stop_mode',type=bool, default=False, help="training the model with early stop mode" )

    args = parser.parse_args()

    train_input = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_input_convlstm.pt").float()

    train_input_mean = torch.mean(train_input, dim=0, keepdim=True).float()
    train_input_std = torch.std(train_input, dim=0, keepdim=True).float()
    train_input_normalization = Normalization(train_input_mean, train_input_std)
    train_input = train_input_normalization.normalize(train_input)

    train_label = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/train_label_convlstm.pt").float()
    train_label_mean = torch.mean(train_label, dim=0, keepdim=False)
    train_label_std = torch.std(train_label, dim=0, keepdim=False)
    train_label_normalization = Normalization(train_label_mean, train_label_std)
    train_label = train_label_normalization.normalize(train_label)

    test_input = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_input_convlstm.pt").float()
    test_input_mean = torch.mean(test_input, dim=0, keepdim=True)
    test_input_std = torch.std(test_input, dim=0, keepdim=True)
    test_input_normalization = Normalization(test_input_mean, test_input_std)
    test_input = test_input_normalization.normalize(test_input)

    test_label = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/test_label_convlstm.pt").float()
    test_label_mean = torch.mean(test_label, dim=0, keepdim=False)
    test_label_std = torch.std(test_label, dim=0, keepdim=False)
    test_label_normalization = Normalization(test_label_mean, test_label_std)
    test_label = test_label_normalization.normalize(test_label)

    valid_input = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_input_convlstm.pt").float()
    valid_input_mean = torch.mean(valid_input, dim=0, keepdim=True)
    valid_input_std = torch.std(valid_input, dim=0, keepdim=True)
    valid_input_normalization = Normalization(valid_input_mean, valid_input_std)
    valid_input = valid_input_normalization.normalize(valid_input)

    valid_label = torch.load(r"/home/0283901/chu1/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data/valid_label_convlstm.pt").float()
    valid_label_mean = torch.mean(valid_label, dim=0, keepdim=False)
    valid_label_std = torch.std(valid_label, dim=0, keepdim=False)
    valid_label_normalization = Normalization(valid_label_mean, valid_label_std)
    valid_label = valid_label_normalization.normalize(valid_label)

    train_loader = get_data(train_input, train_label, args)
    test_loader = get_data(test_input, test_label, args)
    valid_loader = get_data(valid_input, valid_label, args)

    train(train_loader, valid_loader, test_loader, args)
    test(test_loader, args)


    # use the following code if test:
    # parser.set_defaults(test_checkpoint="May16_173759")
    # args = parser.parse_args()
    # test(test_loader, args)





















