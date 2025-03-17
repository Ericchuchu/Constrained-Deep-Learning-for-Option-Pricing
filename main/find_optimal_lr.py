import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset
from torch_lr_finder import LRFinder
from pathlib import Path
from core.models_multi_patch_former_adjusted import MultiPatchFormer
from core.functions import Normalization
from scipy.stats import gaussian_kde

# define Steepest Gradient algorithm
def find_steepest_gradient(log_lrs, losses, smooth_f=0.05, window_size=5):
    # smooth losses
    smooth_losses = []
    for i in range(len(losses)):
        if i == 0:
            smooth_losses.append(losses[i])
        else:
            smooth_losses.append(smooth_f * losses[i] + (1 - smooth_f) * smooth_losses[i-1])
    
    # compute gradients
    gradients = []
    for i in range(1, len(log_lrs)):
        grad = (smooth_losses[i] - smooth_losses[i-1]) / (log_lrs[i] - log_lrs[i-1])
        gradients.append(grad)
    
    # Use rolling window to avoid noises
    avg_gradients = []
    for i in range(len(gradients) - window_size + 1):
        avg_gradients.append(sum(gradients[i:i+window_size]) / window_size)
    
    # find the steepest gradieent
    if len(avg_gradients) > 0:
        steepest_idx = np.argmin(avg_gradients)
        
        # suggest lr : mid point of window
        mid_idx = steepest_idx + window_size // 2
        if mid_idx < len(log_lrs):
            suggested_lr = 10 ** log_lrs[mid_idx]
            return suggested_lr
    
    # default value
    return 1e-3

def find_optimal_lr(model, train_loader, criterion, optimizer_type='adam', 
                   start_lr=1e-7, end_lr=10, num_iter=100, 
                   smooth_f=0.05, window_size=5):

    # initialize optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=start_lr)
    else:
        optimizer = SGD(model.parameters(), lr=start_lr)
    
    # initialize lr finder
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # process lr range test
    print("Processing learning rate range test...")
    lr_finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode="exp",
        smooth_f=smooth_f,
        diverge_th=5.0
    )
    
    # get historical learning rate and losses
    lrs = lr_finder.history['lr']
    losses = lr_finder.history['loss']
    
    # log scaling
    log_lrs = [np.log10(lr) for lr in lrs]
    
    # Use steepest gradient algorithm to find optimal lr
    optimal_lr = find_steepest_gradient(log_lrs, losses, smooth_f, window_size)
    
    # visualize
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.axvline(x=optimal_lr, color='r', linestyle='--', 
                label=f'Optimal LR: {optimal_lr:.6f}')
    plt.legend()
    plt.grid(True)
    plt.title('Learning Rate Finder with Steepest Gradient')
    plt.savefig('optimal_lr_mpf.png')
    plt.show()
    
    # reset model
    lr_finder.reset()
    
    print(f"Suggesting optimal lr : {optimal_lr:.6f}")
    return optimal_lr

class DataProcessor:
    """Handle data loading and preprocessing operations."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.kde_model = None
        
    def compute_train_kde(self, train_input: torch.Tensor):
        """Compute KDE model from training dataset's invm values."""
        # Extract required values from training data
        moneyness = train_input[:, 0, -1, 2].numpy()
                
        # Fit KDE model
        self.kde_model = gaussian_kde(moneyness.reshape(-1, 1).T, bw_method=0.5)
        weights = 1.0 / (self.kde_model(moneyness.reshape(-1, 1).T) + 1e-6)
        weights = torch.tensor(weights)
        mean_weights = weights.mean()
        
        return self.kde_model, mean_weights
        
    def load_and_normalize(self, prefix: str):
        """
        Load and normalize input and label data.
        
        Args:
            prefix: Dataset prefix (train/test/valid)
            
        Returns:
            Normalized input, label, timestamp tensors, and normalization objects
        """
        try:
            # Load data with "mpf" in the filename
            input_tensor = torch.load(self.data_dir / f"{prefix}_input_mpf.pt").float()
            label_tensor = torch.load(self.data_dir / f"{prefix}_label_mpf.pt").float()
            timestamp_tensor = torch.load(self.data_dir / f"{prefix}_label_timestamp_mpf.pt").float()
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

def get_data_loader(input_tensor, data_label, timestamp_tensor, batch_size):
    """Create DataLoader for the given tensors."""
    return DataLoader(
        TensorDataset(input_tensor, data_label, timestamp_tensor),
        batch_size=batch_size,
        shuffle=True,  # Enable shuffling for better training
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device : {device}')
    data_dir = Path("/Users/chuchu/Desktop/option_pricing_thesis_project/3D-Tensor-based-Deep-Learning-Models-for-Predicting-Option-Price-main/data/torch-data")
    
    try:
        processor = DataProcessor(data_dir)
        print("Processing dataset...")
        input_tensor, label_tensor, timestamp_tensor, input_tensor_norm, label_norm = processor.load_and_normalize('train')
        
        # 獲取數據形狀
        B, C, S, F = input_tensor.shape
        print(f"input shape : Batch={B}, Channels={C}, Sequence={S}, Features={F}")
        
        # 創建數據加載器
        batch_size = 64
        train_loader = get_data_loader(input_tensor, label_tensor, timestamp_tensor, batch_size)
        
        # initialize model
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
        
        # 設置損失函數
        criterion = nn.MSELoss()
        
        # 找到最佳學習率
        print("Starting to find optimal learning rate...")
        optimal_lr = find_optimal_lr(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer_type='adam',
            start_lr=1e-9,
            end_lr=1.0,
            num_iter=200,
            smooth_f=0.05,
            window_size=5
        )
        
        print(f"Optimal learning rate : {optimal_lr:.6f}")

    except Exception as e:
        print(f"Error occurence : {str(e)}")
        raise

if __name__ == "__main__":
    main()
