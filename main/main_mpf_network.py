import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import torch.nn as nn 
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_cosine_schedule_with_warmup
from core.models_multi_patch_former_adjusted import MultiPatchFormer
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from core.functions import Normalization, metrics
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

class DataProcessor:
    """Handle data loading and preprocessing operations."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.kde_model = None
        
    def compute_train_kde(self, train_input: torch.Tensor):
        """Compute KDE model from training dataset's invm values."""
        # Extract required values from training data
        time_value = train_input[:, 0, -1, 1].numpy()
                
        # Fit KDE model
        self.kde_model = gaussian_kde(time_value.reshape(-1, 1).T, bw_method=0.5)
        weights = 1.0 / (self.kde_model(time_value.reshape(-1, 1).T) + 1e-6)
        weights = torch.tensor(weights)
        mean_weights = weights.mean()
        
        return self.kde_model, mean_weights
        
    def load_and_normalize(self, prefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Normalization, Normalization]:
        """
        Load and normalize input and label data.
        
        Args:
            prefix: Dataset prefix (train/test/valid)
            
        Returns:
            Normalized input, label, timestamp tensors, and normalization objects
        """
        try:
            # Load data with "mpf" in the filename
            input_tensor = torch.load(self.data_dir / f"{prefix}_input_mpf.pt", weights_only=True).float()
            label_tensor = torch.load(self.data_dir / f"{prefix}_label_mpf.pt", weights_only=True).float()
            timestamp_tensor = torch.load(self.data_dir / f"{prefix}_label_timestamp_mpf.pt", weights_only=True).float()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load data files for prefix '{prefix}': {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading data for prefix '{prefix}': {str(e)}")
        
        # Normalize input with improved epsilon handling
        eps = 1e-8  # Smaller epsilon for better numerical stability
        
        # Normalize input
        input_tensor_mean = torch.mean(input_tensor, dim=0, keepdim=True)
        input_tensor_std = torch.std(input_tensor, dim=0, keepdim=True)
        input_tensor_std = torch.max(input_tensor_std, torch.tensor([eps]))
        input_tensor_norm = Normalization(input_tensor_mean, input_tensor_std)
        input_tensor = input_tensor_norm.normalize(input_tensor)
        
        # Normalize label
        label_mean = torch.mean(label_tensor, dim=0, keepdim=False)
        label_std = torch.std(label_tensor, dim=0, keepdim=False)
        label_std = torch.max(label_std, torch.tensor([eps]))
        label_norm = Normalization(label_mean, label_std)
        label_tensor = label_norm.normalize(label_tensor)
        
        return input_tensor, label_tensor, timestamp_tensor, input_tensor_norm, label_norm

def get_data_loader(input_tensor: torch.Tensor, data_label: torch.Tensor, 
                   timestamp_tensor: torch.Tensor, batch_size: int) -> data.DataLoader:
    """Create DataLoader for the given tensors."""
    return data.DataLoader(
        data.TensorDataset(input_tensor, data_label, timestamp_tensor),
        batch_size=batch_size,
        shuffle=True,  # Enable shuffling for better training
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

def compute_losses(criterion, x_input, input_tensor_norm, output, y, kde_model, mean_weights):
    # For MPF model, we assume output is just the predicted value
    V = output

    # Extract moneyness
    time_value = x_input[:, 0, -1, 1].detach().cpu().numpy()

    # Fit KDE: invm is already a numpy array, so reshape as required
    weights = 1.0 / (kde_model(time_value.reshape(1, -1)) + 1e-6)
    
    # Convert weights back to tensor
    weights = torch.tensor(weights, device=V.device, dtype=V.dtype)
    weights = weights / mean_weights # Normalize weights
    
    # Compute weighted MSE loss
    mse_loss = torch.sum(weights * (V - y) ** 2) / torch.sum(weights)
    
    return mse_loss 

def evaluate(model, data_loader, criterion, dict, device, last_epoch=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_input, y, timestamp in data_loader:
            x_input = x_input.to(device)
            y = y.to(device)
            
            output = model(x_input)
            mse_loss = criterion(output, y)
            loss_value = mse_loss.detach().item()
            total_loss += loss_value

            if last_epoch:
                for ts in timestamp:
                    date_str = f"{int(ts[0]):04d}{int(ts[1]):02d}{int(ts[2]):02d}"
                    if date_str in dict:
                        dict[date_str].append(loss_value)
                    else:
                        dict[date_str] = [loss_value]
    
    return total_loss / len(data_loader), dict

def train_model(model: nn.Module, train_loader: data.DataLoader, valid_loader: data.DataLoader, test_loader: data.DataLoader, input_tensor_norm: Normalization,
                optimizer: torch.optim, num_epochs: int, device: torch.device, args: argparse.ArgumentParser, mean_weights: torch.Tensor, kde_model=None) -> Tuple[dict, dict, dict]:
    # Calculate total training steps for the scheduler
    total_steps = len(train_loader) * num_epochs
    
    # Initialize learning rate scheduler with linear warmup and cosine decay
    # This is more suitable for Transformer-based models like Multi-Patch Former
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),  # Warmup steps based on ratio
        num_training_steps=total_steps
    )
    
    criterion = nn.MSELoss()
    train_loss_dict = {}
    valid_loss_dict = {}
    test_loss_dict = {}
    train_losses = []
    valid_losses = []
    test_losses = []
    
    best_valid_loss = float('inf')
    patience = 6
    trigger_times = 0
    early_stop = False
    minimum_test_loss_epoch = 0
    best_test_loss = float('inf')
    
    # Gradient clipping value
    max_grad_norm = 1.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_losses = []

        for x_input, y, timestamp in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            x_input = x_input.to(device).requires_grad_(True)
            y = y.to(device).requires_grad_(True)
            
            optimizer.zero_grad()
            output = model(x_input)
            # adjusted loss using KDE
            adjusted_loss = compute_losses(criterion, x_input, input_tensor_norm, output, y, kde_model, mean_weights)
            # adjusted_loss = criterion(output, y)
            loss_value = adjusted_loss.detach().item()
            adjusted_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()  # Update scheduler after each batch
            epoch_losses.append(loss_value)

            # Store losses for the last epoch
            if epoch == (num_epochs - 1):
                for ts in timestamp:
                    date_str = f"{int(ts[0]):04d}{int(ts[1]):02d}{int(ts[2]):02d}"
                    if date_str in train_loss_dict:
                        train_loss_dict[date_str].append(loss_value)
                    else:
                        train_loss_dict[date_str] = [loss_value]

        # Calculate average epoch loss
        epoch_loss = np.mean(epoch_losses)
        train_losses.append(epoch_loss)

        # Evaluate on validation and test sets
        valid_loss, valid_loss_dict = evaluate(model, valid_loader, criterion, valid_loss_dict, device, epoch == (num_epochs - 1))
        test_loss, test_loss_dict = evaluate(model, test_loader, criterion, test_loss_dict, device, epoch == (num_epochs - 1))

        valid_losses.append(valid_loss)
        test_losses.append(test_loss)

        # Track best performance
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_loss': valid_loss,
            }, f"checkpoints/mpf_network_best.pth")

        patience_delta = 0.03
        if valid_loss > best_valid_loss * (1 + patience_delta):
            trigger_times += 1
        else:
            trigger_times = 0
                        
        # Track best test performance
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            minimum_test_loss_epoch = epoch

        # Early stopping check
        if args.early_stop_mode and trigger_times >= patience:
            print(f'Early stopping triggered at epoch {epoch}')
            print(f'Best validation loss: {best_valid_loss:.6f}')
            early_stop = True
            break

        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.6f}')
        print(f'Valid Loss: {valid_loss:.6f}')
        print(f'Test Loss: {test_loss:.6f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

    # Plot training curves
    epochs = np.arange(len(train_losses))
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    
    if early_stop:
        plt.axvline(x=len(train_losses)-1, color='r', linestyle='--', label='Early Stopping')
        plt.text(len(train_losses)-1, plt.ylim()[1], 'Early Stop', color='red', horizontalalignment='right')
    
    plt.axvline(x=minimum_test_loss_epoch, color='g', linestyle='--', label='Minimum Test Loss')
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_curves/loss_curves_mpf_network_epoch{args.max_epoch}_early_stop{args.early_stop_mode}_syn{args.syn_data}.png')
    plt.close()

    return train_loss_dict, valid_loss_dict, test_loss_dict

def test_model(model: nn.Module, data_loader: data.DataLoader, dataset_type: str, 
               input_norm: Normalization, label_norm: Normalization, 
               device: torch.device, args: argparse.ArgumentParser) -> pd.DataFrame:
    model.eval()
    criterion = nn.MSELoss()
    
    results = {
        'timestamp': [], 'true_price': [], 'estimated_price': [],
        'moneyness': [], 'time_to_maturity': []
    }
    
    metrics_list = {'loss': [], 'map': [], 'mape': [], 'corr': []}
    
    print(f"\nPredicting results for {dataset_type} data")
    with torch.no_grad():
        for x_input, y, timestamp in tqdm(data_loader, desc=f"Processing {dataset_type}"):
            x_input = x_input.to(device)
            y = y.to(device)
            
            output = model(x_input)
 
            # Unnormalize predictions and true values
            y = label_norm.unnormalize(y)
            y_hat = label_norm.unnormalize(output)
            
            # Calculate metrics
            loss = criterion(y, y_hat)
            corr, map_val, mape = metrics(y_hat, y)
            
            # Store metrics
            metrics_list['loss'].append(loss.item())
            metrics_list['map'].append(map_val.item())
            metrics_list['mape'].append(mape.item())
            metrics_list['corr'].append(corr.item())
            
            # Unnormalize input features
            x_input = input_norm.unnormalize(x_input)
            
            # Store results
            results['timestamp'].extend([''.join(str(t) for t in ts) for ts in timestamp.cpu().numpy()])
            results['true_price'].extend(y.cpu().numpy())
            results['estimated_price'].extend(y_hat.cpu().numpy())
            results['moneyness'].extend(x_input[:,0,-1,2].cpu().numpy())
            results['time_to_maturity'].extend(x_input[:,0,-1,0].cpu().numpy())
    
    # Print average metrics
    print(f"\n{dataset_type.upper()} Metrics:")
    avg_metrics = {k: sum(v) / len(v) for k, v in metrics_list.items()}
    for metric, value in avg_metrics.items():
        print(f"Average {metric.upper()}: {value:.5f}")

    # Enhanced 3D visualization
    plot_len = min(120, len(results['moneyness']))
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize data for better visualization
    moneyness = np.array(results['moneyness'][:plot_len])
    time_to_maturity = np.array(results['time_to_maturity'][:plot_len])
    true_prices = np.array(results['true_price'][:plot_len])
    estimated_prices = np.array(results['estimated_price'][:plot_len])
    
    # Create scatter plots with improved visibility
    scatter1 = ax.scatter(moneyness, time_to_maturity, true_prices, 
                         c='blue', marker='o', label='True Price', alpha=0.6)
    scatter2 = ax.scatter(moneyness, time_to_maturity, estimated_prices, 
                         c='red', marker='^', label='Estimated Price', alpha=0.6)
    
    # Add error lines
    for i in range(plot_len):
        ax.plot([moneyness[i], moneyness[i]], 
                [time_to_maturity[i], time_to_maturity[i]], 
                [true_prices[i], estimated_prices[i]], 
                'k-', alpha=0.2)
    
    # Enhance the plot
    ax.set_title('True vs. Estimated Option Prices', pad=20, size=14)
    ax.set_xlabel('Moneyness', labelpad=10)
    ax.set_ylabel('Time to Maturity', labelpad=10)
    ax.set_zlabel('Option Price', labelpad=10)
    
    # Add grid
    ax.grid(True)
    
    # Adjust the view
    ax.view_init(elev=20, azim=45)
    
    # Add legend
    ax.legend()
    
    # Tight layout
    plt.tight_layout()
    
    # Save the enhanced plot
    plt.savefig(f'testing_result/testing_result_mpf_network_epoch{args.max_epoch}_early_stop{args.early_stop_mode}_syn{args.syn_data}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return pd.DataFrame(results)

def plot_timestamp_loss_modified(train_loss_dict, valid_loss_dict, test_loss_dict, args):
    # Extract test dates
    test_dates = sorted(test_loss_dict.keys())

    # Exclude specific dates
    excluded_dates = ['20210507', '20210514']
    test_dates = [date for date in test_dates if date not in excluded_dates]
    
    # Create dictionaries for loss values
    train_loss_values = {}
    valid_loss_values = {}
    test_loss_values = {}
    
    for test_date in test_dates:
        # Calculate validation loss
        valid_dates = [date for date in valid_loss_dict.keys() if date < test_date][-args.valid_days:]
        valid_losses = [valid_loss_dict[date] for date in valid_dates]
        # Handle nested lists by flattening them
        flattened_valid_losses = []
        for loss_list in valid_losses:
            if isinstance(loss_list, list):
                flattened_valid_losses.extend(loss_list)
            else:
                flattened_valid_losses.append(loss_list)
        valid_loss_values[test_date] = np.mean(flattened_valid_losses) if flattened_valid_losses else 0.0
        
        # Calculate training loss
        train_dates = [date for date in train_loss_dict.keys() if date < test_date][-args.train_days:]
        train_losses = [train_loss_dict[date] for date in train_dates]
        # Handle nested lists by flattening them
        flattened_train_losses = []
        for loss_list in train_losses:
            if isinstance(loss_list, list):
                flattened_train_losses.extend(loss_list)
            else:
                flattened_train_losses.append(loss_list)
        train_loss_values[test_date] = np.mean(flattened_train_losses) if flattened_train_losses else 0.0
        
        # Get test loss and handle potential nested lists
        test_losses = test_loss_dict[test_date]
        if isinstance(test_losses, list):
            test_loss_values[test_date] = np.mean(test_losses)
        else:
            test_loss_values[test_date] = test_losses
    
    # Create enhanced visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot with improved styling
    ax.plot(test_dates, [train_loss_values[date] for date in test_dates], 
            marker='o', linestyle='-', linewidth=2, color='#2E86C1', 
            label=f'Avg Train Loss ({args.train_days} days)')
    ax.plot(test_dates, [valid_loss_values[date] for date in test_dates], 
            marker='s', linestyle='-', linewidth=2, color='#28B463', 
            label=f'Validation Loss ({args.valid_days} day before)')
    ax.plot(test_dates, [test_loss_values[date] for date in test_dates], 
            marker='^', linestyle='-', linewidth=2, color='#E74C3C', 
            label='Test Loss')

    # Enhance the plot
    ax.set_title('Training, Validation, and Test Loss Over Time', pad=20, size=14)
    ax.set_xlabel('Date', size=12)
    ax.set_ylabel('Loss Value', size=12)
    
    # Rotate and align the tick labels so they look better
    ax.set_xticks(test_dates)
    ax.set_xticklabels(test_dates, rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with better positioning
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the enhanced plot
    plt.savefig(f'loss_curves/timestamp_loss_curves_mpf_network_epoch{args.max_epoch}_early_stop{args.early_stop_mode}_syn{args.syn_data}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-learning_rate', type=float, default= 1e-4, help="learning rate of the Adam")
    parser.add_argument('-max_epoch', type=int, default=100, help="maximum number of training epochs")
    parser.add_argument('-batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'),
                        help="name of the session to be used in saving the model")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None,
                        help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-nonlinearity', action="store", type=str, default="tanh",
                        help="Type of nonlinearity for the CNN [tanh, relu]", choices=["tanh", "relu"])
    parser.add_argument('-early_stop_mode', action='store_true', help="training the model with early stop mode")  
    parser.add_argument('-train_days', type=int, default=3, help="the days for training")
    parser.add_argument('-valid_days', type=int, default=1, help="the days for validating")
    parser.add_argument('-test_days', type=int, default=1, help="the days for testing")
    parser.add_argument('-syn_data', type=bool, default=True, help="synthesize the missing data from TAIEX website")
    
    # Learning rate scheduler parameters for cosine scheduler with warmup
    parser.add_argument('-warmup_ratio', type=float, default=0.1, help="Ratio of total training steps to use for warmup")
    parser.add_argument('-min_lr', type=float, default=1e-6, help="Minimum learning rate at the end of the schedule")

    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps")
    print(f'Using device: {device}')
    
    # Set data directory
    data_dir = Path("/Users/chuchu/Desktop/option_pricing_thesis_project/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data")
    
    try:
        # Initialize data processor
        processor = DataProcessor(data_dir)
        
        # Load and process datasets
        datasets = {}
        normalizations = {}
        raw_data = {}
        for dataset_type in ['train', 'test', 'valid']:
            input_tensor, label_tensor, timestamp_tensor, input_tensor_norm, label_norm = processor.load_and_normalize(dataset_type)
            datasets[dataset_type] = get_data_loader(input_tensor, label_tensor, timestamp_tensor, args.batch_size)
            normalizations[dataset_type] = (input_tensor_norm, label_norm)
            raw_data[dataset_type] = (input_tensor, label_tensor, timestamp_tensor)
        
        # Compute KDE model from training data
        print("Computing KDE model from training data...")
        kde_model, mean_weights = processor.compute_train_kde(normalizations['train'][0].unnormalize(raw_data['train'][0]))

        # Initialize model
        # Assuming input shape is (batch_size, channels, sequence, features)
        # Get feature shape from the data
        _, C, S, F = raw_data['train'][0].shape
        
        model = MultiPatchFormer(
            patch_configs=[(2, 1), (3,1), (4,1), (5, 1)],  # Example patch configs for sequence
            feature_shape=F,
            embed_dim=128,
            in_channels=C,
            temporal_layers=3,
            temporal_heads=4,
            channel_heads=4,
            decoder_patch_length=2,
            decoder_steps=4,
            decoder_hidden_dim=128,
            decoder_out_dim=1,  # Assuming single output value
            dropout=0.1
        ).to(device)
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Train or load model
        if args.test_checkpoint:
            # Load pre-trained model
            checkpoint = torch.load(args.test_checkpoint, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {args.test_checkpoint}")
        else:
            # Train model
            train_loss_dict, valid_loss_dict, test_loss_dict = train_model(
                model, datasets['train'], datasets['valid'], datasets['test'], normalizations['train'][0],
                optimizer, args.max_epoch, device, args, mean_weights, kde_model
            )
            plot_timestamp_loss_modified(train_loss_dict, valid_loss_dict, test_loss_dict, args)

        # Load model with minimum valid loss
        # Load the checkpoint
        checkpoint = torch.load(f"checkpoints/mpf_network_best.pth", map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # Evaluate model
        test_results_df = test_model(
            model, datasets['test'], 'test',
            normalizations['test'][0], normalizations['test'][1],
            device, args
        )
        
        # Save results
        timestamp = datetime.now().strftime('%b%d_%H%M%S')
        results_path = f'{timestamp}_mpf_network_results.csv'
        test_results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
