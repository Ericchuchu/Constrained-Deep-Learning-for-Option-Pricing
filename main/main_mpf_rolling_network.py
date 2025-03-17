import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import torch.nn as nn 
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
from core.models_multi_patch_former_adjusted import MultiPatchFormer
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from core.functions import Normalization, metrics
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import copy

class DataProcessor:
    """Handle data loading and preprocessing operations for rolling window data."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.kde_models = {}  # Store KDE models for each window
        
    def load_rolling_windows_data(self):
        """Load the rolling windows data."""
        try:
            rolling_windows_data = torch.load(self.data_dir / "rolling_windows_data.pt", weights_only=False)
            print(f"Loaded {len(rolling_windows_data)} rolling windows")
            return rolling_windows_data
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load rolling windows data: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading rolling windows data: {str(e)}")
    
    def compute_kde_for_window(self, window_data):
        """Compute KDE model for a specific window's training data."""
        # Extract moneyness values from training data
        train_input = window_data['train_input']
        moneyness = train_input[:, 0, -1, 2].numpy()
                
        # Fit KDE model
        kde_model = gaussian_kde(moneyness.reshape(-1, 1).T, bw_method=0.5)
        weights = 1.0 / (kde_model(moneyness.reshape(-1, 1).T) + 1e-6)
        weights = torch.tensor(weights)
        mean_weights = weights.mean()
        
        return kde_model, mean_weights
        
    def normalize_window_data(self, window_data):
        """Normalize input and label data for a specific window."""
        # Create copies to avoid modifying the original data
        normalized_window = copy.deepcopy(window_data)
        
        # Normalize input tensors
        eps = 1e-8
        
        # Process train data
        train_input = window_data['train_input'].float()
        train_label = window_data['train_label'].float()
        
        # Compute statistics on training data
        input_tensor_mean = torch.mean(train_input, dim=0, keepdim=True)
        input_tensor_std = torch.std(train_input, dim=0, keepdim=True)
        input_tensor_std = torch.max(input_tensor_std, torch.tensor([eps]))
        input_tensor_norm = Normalization(input_tensor_mean, input_tensor_std)
        
        label_mean = torch.mean(train_label, dim=0, keepdim=False)
        label_std = torch.std(train_label, dim=0, keepdim=False)
        label_std = torch.max(label_std, torch.tensor([eps]))
        label_norm = Normalization(label_mean, label_std)
        
        # Normalize all datasets using training statistics
        normalized_window['train_input'] = input_tensor_norm.normalize(train_input)
        normalized_window['train_label'] = label_norm.normalize(train_label)
        
        normalized_window['valid_input'] = input_tensor_norm.normalize(window_data['valid_input'].float())
        normalized_window['valid_label'] = label_norm.normalize(window_data['valid_label'].float())
        
        normalized_window['test_input'] = input_tensor_norm.normalize(window_data['test_input'].float())
        normalized_window['test_label'] = label_norm.normalize(window_data['test_label'].float())
        
        # Store normalization objects
        normalized_window['input_norm'] = input_tensor_norm
        normalized_window['label_norm'] = label_norm
        
        return normalized_window

def get_data_loader(input_tensor: torch.Tensor, data_label: torch.Tensor, 
                   timestamp_tensor: torch.Tensor, batch_size: int, shuffle=True) -> data.DataLoader:
    """Create DataLoader for the given tensors."""
    return data.DataLoader(
        data.TensorDataset(input_tensor, data_label, timestamp_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

def compute_losses(criterion, x_input, input_tensor_norm, output, y, kde_model):
    """Compute weighted loss using KDE model."""
    # For MPF model, we assume output is just the predicted value
    V = output

    # Extract moneyness
    moneyness = x_input[:, 0, -1, 2].detach().cpu().numpy()

    # Fit KDE: invm is already a numpy array, so reshape as required
    weights = 1.0 / (kde_model(moneyness.reshape(1, -1)) + 1e-6)
    
    # Convert weights back to tensor
    weights = torch.tensor(weights, device=V.device, dtype=V.dtype)
    weights = weights / weights.sum()  # Normalize weights
    
    # Compute weighted MSE loss
    mse_loss = torch.mean(weights * (V - y) ** 2)
    
    return mse_loss 

def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_timestamps = []
    
    with torch.no_grad():
        for x_input, y, timestamp in data_loader:
            x_input = x_input.to(device)
            y = y.to(device)
            
            output = model(x_input)
            mse_loss = criterion(output, y)
            loss_value = mse_loss.detach().item()
            total_loss += loss_value
            
            # Store predictions and targets for metrics calculation
            all_predictions.append(output.cpu())
            all_targets.append(y.cpu())
            all_timestamps.append(timestamp)
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_timestamps = torch.cat(all_timestamps, dim=0)
    
    # Calculate metrics
    corr, map_val, mape = metrics(all_predictions, all_targets)
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss, corr.item(), map_val.item(), mape.item(), all_predictions, all_targets, all_timestamps

def train_window_model(model, window_data, optimizer, device, args, kde_model):
    """Train model on a specific window."""
    # Create data loaders
    train_loader = get_data_loader(
        window_data['train_input'], 
        window_data['train_label'], 
        window_data['train_timestamp'], 
        args.batch_size
    )
    
    valid_loader = get_data_loader(
        window_data['valid_input'], 
        window_data['valid_label'], 
        window_data['valid_timestamp'], 
        args.batch_size
    )
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        threshold_mode='rel',
        cooldown=args.lr_cooldown,
        min_lr=args.min_lr,
        eps=1e-8
    )
    
    criterion = nn.MSELoss()
    train_losses = []
    valid_losses = []
    
    best_valid_loss = float('inf')
    patience = 8
    trigger_times = 0
    early_stop = False
    best_model_state = None
    
    # Gradient clipping value
    max_grad_norm = 1.0
    
    for epoch in range(args.max_epoch):
        # Training phase
        model.train()
        epoch_losses = []

        for x_input, y, timestamp in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.max_epoch} - Training', leave=False):
            x_input = x_input.to(device).requires_grad_(True)
            y = y.to(device).requires_grad_(True)
            
            optimizer.zero_grad()
            output = model(x_input)
            adjusted_loss = compute_losses(criterion, x_input, window_data['input_norm'], output, y, kde_model)
            loss_value = adjusted_loss.detach().item()
            adjusted_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            epoch_losses.append(loss_value)

        # Calculate average epoch loss
        epoch_loss = np.mean(epoch_losses)
        train_losses.append(epoch_loss)

        # Evaluate on validation set
        valid_loss, valid_corr, valid_map, valid_mape, _, _, _ = evaluate(model, valid_loader, criterion, device)
        valid_losses.append(valid_loss)

        # Update learning rate scheduler
        scheduler.step(valid_loss)

        # Track best performance and save model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = copy.deepcopy(model.state_dict())
            trigger_times = 0
        else:
            trigger_times += 1
            
        # Early stopping check
        if args.early_stop_mode and trigger_times >= patience:
            print(f'Early stopping triggered at epoch {epoch}')
            print(f'Best validation loss: {best_valid_loss:.6f}')
            early_stop = True
            break

        # Print epoch statistics
        if epoch % 5 == 0 or epoch == args.max_epoch - 1:
            print(f'Epoch {epoch+1}/{args.max_epoch}:')
            print(f'Train Loss: {epoch_loss:.6f}')
            print(f'Valid Loss: {valid_loss:.6f} (Corr: {valid_corr:.4f}, MAP: {valid_map:.4f}, MAPE: {valid_mape:.4f})')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, valid_losses, early_stop

def ensemble_predict(models, weights, x_input, device):
    """ensemble model prediction using weighted sum"""
    predictions = []
    x_input = x_input.to(device)
    
    # 獲取每個模型的預測
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(x_input)
            predictions.append(pred)
    
    # 計算加權平均
    weighted_sum = torch.zeros_like(predictions[0])
    weight_sum = sum(weights)
    
    for i, pred in enumerate(predictions):
        weighted_sum += (weights[i] / weight_sum) * pred
    
    return weighted_sum

def adaptive_ensemble(all_models, window_data, device):
    """adative ensemble：adjust weights based on models performance"""
    # 創建驗證數據加載器
    valid_loader = get_data_loader(
        window_data['valid_input'], 
        window_data['valid_label'], 
        window_data['valid_timestamp'], 
        batch_size=64,
        shuffle=False
    )
    
    # 評估每個模型在驗證集上的性能
    model_performances = []
    
    for model_idx, model in enumerate(all_models):
        model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for x_input, y, _ in valid_loader:
                x_input = x_input.to(device)
                y = y.to(device)
                
                output = model(x_input)
                loss = nn.MSELoss()(output, y)
                valid_loss += loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        model_performances.append((model_idx, avg_valid_loss))
    
    # 根據性能排序模型
    model_performances.sort(key=lambda x: x[1])
    
    # 選擇表現最好的幾個模型
    top_k = min(2, len(model_performances))
    selected_models = [all_models[idx] for idx, _ in model_performances[:top_k]]
    
    # 計算權重（基於驗證損失的倒數）
    weights = [1.0 / (perf + 1e-6) for _, perf in model_performances[:top_k]]
    
    return selected_models, weights

def test_window_model_ensemble(models, weights, window_data, device, window_idx):
    """Test model on a specific window's test set."""
    # Create test data loader
    test_loader = get_data_loader(
        window_data['test_input'], 
        window_data['test_label'], 
        window_data['test_timestamp'], 
        batch_size=64,
        shuffle=False  # No need to shuffle test data
    )
    
    criterion = nn.MSELoss()

    # evaluate
    all_predictions = []
    all_targets = []
    all_timestamps = []
    
    for x_input, y, timestamp in test_loader:
        x_input = x_input.to(device)
        y = y.to(device)
        
        # 使用集成預測
        output = ensemble_predict(models, weights, x_input, device)
        
        # 儲存結果
        all_predictions.append(output.cpu())
        all_targets.append(y.cpu())
        all_timestamps.append(timestamp)
    
    # 合併批次結果
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_timestamps = torch.cat(all_timestamps, dim=0)
    
    # 計算指標
    mse_loss = torch.mean((all_predictions - all_targets) ** 2).item()
    corr, map_val, mape = metrics(all_predictions, all_targets)
    
    # Extract feature from input
    moneyness = window_data['test_input'][:,0,-1,2]
    time_to_maturity = window_data['test_input'][:,0,-1,0]
    
    # Unnormalize predictions and targets
    predictions_unnorm = window_data['label_norm'].unnormalize(all_predictions)
    targets_unnorm = window_data['label_norm'].unnormalize(all_targets)
    
    # Create results dataframe
    results = {
        'window_idx': [window_idx] * len(all_predictions),
        'timestamp': [''.join(str(t) for t in ts) for ts in all_timestamps.numpy()],
        'true_price': targets_unnorm.numpy(),
        'estimated_price': predictions_unnorm.numpy(),
        'moneyness':moneyness.numpy(),
        'time_to_maturity': time_to_maturity.numpy()
    }
    
    results_df = pd.DataFrame(results)
    
    # Print test metrics
    print(f"\nWindow {window_idx} Test Results:")
    print(f"Loss: {mse_loss:.6f}")
    print(f"Correlation: {corr.item():.6f}")
    print(f"MAP: {map_val.item():.6f}")
    print(f"MAPE: {mape.item():.6f}")
    
    return results_df, mse_loss, corr, map_val, mape

def plot_window_results(window_metrics, args):
    """Plot metrics across all windows."""
    windows = list(range(len(window_metrics)))
    
    # Extract metrics
    losses = [metrics['test_loss'] for metrics in window_metrics]
    corrs = [metrics['test_corr'] for metrics in window_metrics]
    maps = [metrics['test_map'] for metrics in window_metrics]
    mapes = [metrics['test_mape'] for metrics in window_metrics]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axs[0, 0].plot(windows, losses, marker='o', linestyle='-', linewidth=2, color='#E74C3C')
    axs[0, 0].set_title('Test Loss by Window', size=12)
    axs[0, 0].set_xlabel('Window Index', size=10)
    axs[0, 0].set_ylabel('Loss', size=10)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot correlation
    axs[0, 1].plot(windows, corrs, marker='o', linestyle='-', linewidth=2, color='#2E86C1')
    axs[0, 1].set_title('Test Correlation by Window', size=12)
    axs[0, 1].set_xlabel('Window Index', size=10)
    axs[0, 1].set_ylabel('Correlation', size=10)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot MAP
    axs[1, 0].plot(windows, maps, marker='o', linestyle='-', linewidth=2, color='#28B463')
    axs[1, 0].set_title('Test MAP by Window', size=12)
    axs[1, 0].set_xlabel('Window Index', size=10)
    axs[1, 0].set_ylabel('MAP', size=10)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot MAPE
    axs[1, 1].plot(windows, mapes, marker='o', linestyle='-', linewidth=2, color='#F39C12')
    axs[1, 1].set_title('Test MAPE by Window', size=12)
    axs[1, 1].set_xlabel('Window Index', size=10)
    axs[1, 1].set_ylabel('MAPE', size=10)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%b%d_%H%M%S')
    plt.savefig(f'testing_result/rolling_windows/rolling_window_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cumulative_performance(all_results_df, args):
    """Plot cumulative performance metrics."""
    # Group by window index
    window_groups = all_results_df.groupby('window_idx')
    
    # Calculate metrics for each window
    window_metrics = []
    for window_idx, group in window_groups:
        true_prices = group['true_price'].values
        estimated_prices = group['estimated_price'].values
        
        # Convert to tensors for metrics calculation
        true_tensor = torch.tensor(true_prices)
        est_tensor = torch.tensor(estimated_prices)
        
        # Calculate metrics
        corr, map_val, mape = metrics(est_tensor, true_tensor)
        mse = ((est_tensor - true_tensor) ** 2).mean().item()
        
        window_metrics.append({
            'window_idx': window_idx,
            'mse': mse,
            'corr': corr.item(),
            'map': map_val.item(),
            'mape': mape.item()
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(window_metrics)
    
    # Calculate cumulative metrics
    cumulative_metrics = []
    for i in range(len(metrics_df)):
        subset = metrics_df.iloc[:i+1]
        cumulative_metrics.append({
            'window_idx': i,
            'cum_mse': subset['mse'].mean(),
            'cum_corr': subset['corr'].mean(),
            'cum_map': subset['map'].mean(),
            'cum_mape': subset['mape'].mean()
        })
    
    cum_df = pd.DataFrame(cumulative_metrics)
    
    # Plot cumulative metrics
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot MSE
    axs[0, 0].plot(cum_df['window_idx'], cum_df['cum_mse'], marker='o', linestyle='-', linewidth=2, color='#E74C3C')
    axs[0, 0].set_title('Cumulative MSE', size=12)
    axs[0, 0].set_xlabel('Window Index', size=10)
    axs[0, 0].set_ylabel('MSE', size=10)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot correlation
    axs[0, 1].plot(cum_df['window_idx'], cum_df['cum_corr'], marker='o', linestyle='-', linewidth=2, color='#2E86C1')
    axs[0, 1].set_title('Cumulative Correlation', size=12)
    axs[0, 1].set_xlabel('Window Index', size=10)
    axs[0, 1].set_ylabel('Correlation', size=10)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot MAP
    axs[1, 0].plot(cum_df['window_idx'], cum_df['cum_map'], marker='o', linestyle='-', linewidth=2, color='#28B463')
    axs[1, 0].set_title('Cumulative MAP', size=12)
    axs[1, 0].set_xlabel('Window Index', size=10)
    axs[1, 0].set_ylabel('MAP', size=10)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot MAPE
    axs[1, 1].plot(cum_df['window_idx'], cum_df['cum_mape'], marker='o', linestyle='-', linewidth=2, color='#F39C12')
    axs[1, 1].set_title('Cumulative MAPE', size=12)
    axs[1, 1].set_xlabel('Window Index', size=10)
    axs[1, 1].set_ylabel('MAPE', size=10)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%b%d_%H%M%S')
    plt.savefig(f'testing_result/rolling_windows/cumulative_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return cum_df

def visualize_predictions(all_results_df, args):
    """Create 3D visualization of predictions vs true values using moneyness and time to maturity as axes."""
    # Sample data for visualization (limit to avoid overcrowding)
    sample_size = min(500, len(all_results_df))
    sample_df = all_results_df.sample(sample_size)
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract data
    moneyness = sample_df['moneyness'].values
    time_to_maturity = sample_df['time_to_maturity'].values
    true_prices = sample_df['true_price'].values
    estimated_prices = sample_df['estimated_price'].values
    
    # Create scatter plots for both true and estimated prices
    scatter1 = ax.scatter(moneyness, time_to_maturity, true_prices, 
                         c='blue', marker='o', label='True Prices', alpha=0.6)
    scatter2 = ax.scatter(moneyness, time_to_maturity, estimated_prices, 
                         c='red', marker='^', label='Estimated Prices', alpha=0.6)
    
    # Add error lines connecting true and estimated prices
    for i in range(len(moneyness)):
        ax.plot([moneyness[i], moneyness[i]], 
                [time_to_maturity[i], time_to_maturity[i]], 
                [true_prices[i], estimated_prices[i]], 
                'k-', alpha=0.2)
    
    # Add labels and title
    ax.set_title('True vs. Estimated Option Prices', pad=20, size=14)
    ax.set_xlabel('Moneyness', labelpad=10)
    ax.set_ylabel('Time to Maturity', labelpad=10)
    ax.set_zlabel('Option Price', labelpad=10)
    
    # Add grid
    ax.grid(True)
    
    # Adjust the view
    ax.view_init(elev=30, azim=45)
    
    # Add legend
    ax.legend()
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%b%d_%H%M%S')
    plt.savefig(f'testing_result/rolling_windows/3d_predictions_moneyness_ttm_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-learning_rate', type=float, default=0.0001, help="learning rate of the Adam")
    parser.add_argument('-max_epoch', type=int, default=50, help="maximum number of training epochs per window")
    parser.add_argument('-batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'),
                        help="name of the session to be used in saving the model")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None,
                        help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-nonlinearity', action="store", type=str, default="tanh",
                        help="Type of nonlinearity for the CNN [tanh, relu]", choices=["tanh", "relu"])
    parser.add_argument('-early_stop_mode', type=bool, default=True, help="training the model with early stop mode")
    parser.add_argument('-train_days', type=int, default=3, help="the days for training")
    parser.add_argument('-valid_days', type=int, default=1, help="the days for validating")
    parser.add_argument('-test_days', type=int, default=1, help="the days for testing")
    parser.add_argument('-transfer_learning', type=bool, default=True, 
                        help="use previous window's model as starting point for next window")
    
    # Learning rate scheduler parameters
    parser.add_argument('-lr_factor', type=float, default=0.5, help="Factor by which the learning rate will be reduced")
    parser.add_argument('-lr_patience', type=int, default=3, help="Number of epochs with no improvement after which learning rate will be reduced")
    parser.add_argument('-lr_threshold', type=float, default=0.0001, help="Threshold for measuring the new optimum")
    parser.add_argument('-lr_cooldown', type=int, default=0, help="Number of epochs to wait before resuming normal operation after lr has been reduced")
    parser.add_argument('-min_lr', type=float, default=1e-9, help="Lower bound on the learning rate")

    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('testing_result/rolling_windows', exist_ok=True)
    os.makedirs('loss_curves', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set data directory
    data_dir = Path("/Users/chuchu/Desktop/option_pricing_thesis_project/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data")
    
    try:
        # Initialize data processor
        processor = DataProcessor(data_dir)
        
        # Load rolling windows data
        rolling_windows_data = processor.load_rolling_windows_data()
        
        # Store results for all windows
        all_results_df = pd.DataFrame()
        window_metrics = []
        
        # Initialize model
        # Get feature shape from the first window's data
        first_window = rolling_windows_data[0]
        _, C, S, F = first_window['train_input'].shape
        
        model = MultiPatchFormer(
            patch_configs=[(2, 1), (3, 1)],  # Example patch configs for sequence
            feature_shape=F,
            embed_dim=32,
            in_channels=C,
            temporal_layers=2,
            temporal_heads=4,
            channel_heads=4,
            decoder_patch_length=2,
            decoder_steps=4,
            decoder_hidden_dim=128,
            decoder_out_dim=1,  # Assuming single output value
            dropout=0.1
        ).to(device)
        
        all_window_models = [] # store models for all windows
        # Process each window
        for window_idx, window_data in enumerate(rolling_windows_data):
            print(f"\n{'='*50}")
            print(f"Processing Window {window_idx}")
            print(f"Train period: {window_data['train_start_date']} to {window_data['train_end_date']}")
            print(f"Valid period: {window_data['valid_start_date']} to {window_data['valid_end_date']}")
            print(f"Test period: {window_data['test_start_date']} to {window_data['test_end_date']}")
            print(f"{'='*50}")
            
            # Normalize window data
            normalized_window = processor.normalize_window_data(window_data)
            
            # Compute KDE model for this window
            kde_model, mean_weights = processor.compute_kde_for_window(normalized_window)
            
            # If not using transfer learning or it's the first window, initialize a new model
            if not args.transfer_learning or window_idx == 0:
                if window_idx > 0:
                    print("Initializing new model for this window (no transfer learning)")
                
                model = MultiPatchFormer(
                    patch_configs=[(2, 1), (2,3), (3, 1), (3,2)],
                    feature_shape=F,
                    embed_dim=16,
                    in_channels=C,
                    temporal_layers=2,
                    temporal_heads=4,
                    channel_heads=4,
                    decoder_patch_length=2,
                    decoder_steps=4,
                    decoder_hidden_dim=128,
                    decoder_out_dim=1,
                    dropout=0.1
                ).to(device)
            else:
                print("Using previous window's model as starting point (transfer learning)")
            
            # Initialize optimizer for this window
            # If it's the first window, use the default learning rate
            # Otherwise, use the learning rate from the previous window's optimizer
            if window_idx == 0:
                current_lr = args.learning_rate
            else:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Using learning rate from previous window: {current_lr:.6f}")
            
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            
            # Train model on this window
            model, train_losses, valid_losses, early_stop = train_window_model(
                model, normalized_window, optimizer, device, args, kde_model
            )

            # The model after training
            trained_model = copy.deepcopy(model)
            all_window_models.append(trained_model)
            
            # adaptive ensemble
            selected_models, weights = adaptive_ensemble(all_window_models, window_data, device)

            # Save model for this window
            torch.save({
                'window_idx': window_idx,
                'model_state_dict': model.state_dict(),
                'train_start_date': window_data['train_start_date'],
                'train_end_date': window_data['train_end_date'],
                'test_start_date': window_data['test_start_date'],
                'test_end_date': window_data['test_end_date'],
            }, f"checkpoints/mpf_rolling_window_{window_idx}.pth")
            
            # Test model on this window
            window_results_df, test_loss, test_corr, test_map, test_mape = test_window_model_ensemble(
                selected_models, weights, normalized_window, device, window_idx
            )
            
            # Store window metrics
            window_metrics.append({
                'window_idx': window_idx,
                'train_start_date': window_data['train_start_date'],
                'train_end_date': window_data['train_end_date'],
                'test_start_date': window_data['test_start_date'],
                'test_end_date': window_data['test_end_date'],
                'test_loss': test_loss,
                'test_corr': test_corr,
                'test_map': test_map,
                'test_mape': test_mape,
                'early_stop': early_stop
            })
            
            # Append results to all_results_df
            all_results_df = pd.concat([all_results_df, window_results_df], ignore_index=True)
            
            # Plot training curves for this window
            plt.figure(figsize=(10, 6))
            epochs = np.arange(len(train_losses))
            plt.plot(epochs, train_losses, label='Train Loss')
            plt.plot(epochs, valid_losses, label='Validation Loss')
            
            if early_stop:
                plt.axvline(x=len(train_losses)-1, color='r', linestyle='--', label='Early Stopping')
            
            plt.title(f'Window {window_idx} Training Curves', size=12)
            plt.xlabel('Epoch', size=10)
            plt.ylabel('Loss', size=10)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f'loss_curves/window_{window_idx}_loss_curves_mpf.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot overall window results
        plot_window_results(window_metrics, args)
        
        # Plot cumulative performance
        cum_metrics_df = plot_cumulative_performance(all_results_df, args)
        
        # Visualize predictions in 3D
        visualize_predictions(all_results_df, args)
        
        # Save all results
        timestamp = datetime.now().strftime('%b%d_%H%M%S')
        all_results_df.to_csv(f'testing_result/rolling_windows/all_window_results_{timestamp}.csv', index=False)
        
        # Save window metrics
        metrics_df = pd.DataFrame(window_metrics)
        metrics_df.to_csv(f'testing_result/rolling_windows/window_metrics_{timestamp}.csv', index=False)
        
        # Print final summary
        print("\n" + "="*50)
        print("Rolling Window Training Complete")
        print(f"Total Windows Processed: {len(window_metrics)}")
        print(f"Results saved to testing_result/rolling_windows/all_window_results_{timestamp}.csv")
        print(f"Window metrics saved to testing_result/rolling_windows/window_metrics_{timestamp}.csv")
        print("="*50)

        # total test metrics
        y_true = all_results_df['true_price'].values
        y_pred = all_results_df['estimated_price'].values

        mse_loss = np.mean((y_true - y_pred) ** 2)
        corr, map, mape = metrics(y_true, y_pred)
        print("\n" + "="*50)
        print("TEST metrics")
        print(f"MSE Loss: {mse_loss:.6f}")
        print(f"Correlation: {corr.item():.6f}")
        print(f"MAP: {map.item():.6f}")
        print(f"MAPE: {mape.item():.6f}")
        print("="*50)


    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
