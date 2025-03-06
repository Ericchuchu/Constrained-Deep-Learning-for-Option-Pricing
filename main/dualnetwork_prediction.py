import argparse
import torch
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
import pandas as pd
from tqdm import tqdm
import yaml

base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(base_dir))

from core.models_dual_network import DualBranchNetwork
from core.functions import Normalization, metrics
from torch.utils import data
from torch import nn

# Set for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DataProcessor:
    """Handle data loading and preprocessing operations."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        
    def load_and_normalize(self, prefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Normalization, Normalization]:
        """
        Load and normalize input and label data.
        
        Args:
            prefix: Dataset prefix (train/test/valid)
            
        Returns:
            Normalized input, label, timestamp tensors, and normalization objects
        """
        try:
            # Load data
            input_tensor_convlstm = torch.load(self.data_dir / f"{prefix}_input_convlstm_dual.pt", weights_only=True).float()
            input_tensor_transformer = torch.load(self.data_dir / f"{prefix}_input_transformer_dual.pt", weights_only=True).float()
            label_tensor = torch.load(self.data_dir / f"{prefix}_label_dual.pt", weights_only=True).float()
            timestamp_tensor = torch.load(self.data_dir / f"{prefix}_label_timestamp_dual.pt", weights_only=True).float()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load data files for prefix '{prefix}': {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading data for prefix '{prefix}': {str(e)}")
        
        # Normalize input with improved epsilon handling
        eps = 1e-8  # Smaller epsilon for better numerical stability
        
        # Normalize ConvLSTM input
        input_tensor_convlstm_mean = torch.mean(input_tensor_convlstm, dim=0, keepdim=True)
        input_tensor_convlstm_std = torch.std(input_tensor_convlstm, dim=0, keepdim=True)
        input_tensor_convlstm_std = torch.max(input_tensor_convlstm_std, torch.tensor([eps]))
        input_tensor_convlstm_norm = Normalization(input_tensor_convlstm_mean, input_tensor_convlstm_std)
        input_tensor_convlstm = input_tensor_convlstm_norm.normalize(input_tensor_convlstm)

        # Normalize Transformer input
        input_tensor_transformer_mean = torch.mean(input_tensor_transformer, dim=0, keepdim=True)
        input_tensor_transformer_std = torch.std(input_tensor_transformer, dim=0, keepdim=True)
        input_tensor_transformer_std = torch.max(input_tensor_transformer_std, torch.tensor([eps]))
        input_tensor_transformer_norm = Normalization(input_tensor_transformer_mean, input_tensor_transformer_std)
        input_tensor_transformer = input_tensor_transformer_norm.normalize(input_tensor_transformer)
        
        # Normalize label
        label_mean = torch.mean(label_tensor, dim=0, keepdim=False)
        label_std = torch.std(label_tensor, dim=0, keepdim=False)
        label_std = torch.max(label_std, torch.tensor([eps]))
        label_norm = Normalization(label_mean, label_std)
        label_tensor = label_norm.normalize(label_tensor)
        
        return input_tensor_convlstm, input_tensor_transformer, label_tensor, timestamp_tensor, input_tensor_convlstm_norm, input_tensor_transformer_norm, label_norm

def get_data_loader(convlstm_input: torch.Tensor, transformer_input: torch.Tensor, data_label: torch.Tensor, 
                   timestamp_tensor: torch.Tensor, batch_size: int) -> data.DataLoader:
    """Create DataLoader for the given tensors."""
    return data.DataLoader(
        data.TensorDataset(convlstm_input, transformer_input, data_label, timestamp_tensor),
        batch_size=batch_size,
        shuffle=True,  # Enable shuffling for better training
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device to load model to
        
    Returns:
        Loaded model
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # Initialize model and load state dict
        model = DualBranchNetwork()
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        
        print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['valid_loss']}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

def predict_dataset(model: nn.Module, data_loader: data.DataLoader, dataset_type: str, 
                    convlstm_input_norm: Normalization, transformer_input_norm: Normalization,
                    label_norm: Normalization, device: torch.device, args: argparse.ArgumentParser) -> pd.DataFrame:
    """
    Generate predictions for a dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader containing dataset
        dataset_type: Type of dataset (train/test/valid)
        input_norm: Input normalization object
        label_norm: Label normalization object
        device: torch device to use
        
    Returns:
        DataFrame containing predictions and metrics
    """
    model.eval()
    criterion = nn.MSELoss()
    
    results = {
        'timestamp': [], 'true_price': [], 'estimated_price': [],
        'strike_price': [], 'time_to_maturity': [], 
        'volume': [], 'underlying_price': [], 'cp_flag': []
    }
    
    metrics_list = {'loss': [], 'map': [], 'mape': [], 'corr': []}
    
    print(f"\nPredicting results for {dataset_type} data")
    with torch.no_grad():
        for x_convlstm, x_transformer, y, timestamp in tqdm(data_loader, desc=f"Processing {dataset_type}"):
            x_convlstm = x_convlstm.to(device)
            x_transformer = x_transformer.to(device)
            y = y.to(device)
            
            outputs = model(x_convlstm, x_transformer)
 
            # Unnormalize predictions and true values
            y = label_norm.unnormalize(y)
            y_hat = label_norm.unnormalize(outputs[0])
            
            # Calculate metrics
            loss = criterion(y, y_hat)
            corr, map_val, mape = metrics(y_hat, y)
            
            # Store metrics
            metrics_list['loss'].append(loss.item())
            metrics_list['map'].append(map_val.item())
            metrics_list['mape'].append(mape.item())
            metrics_list['corr'].append(corr.item())
            
            # Unnormalize input features
            x_convlstm = convlstm_input_norm.unnormalize(x_convlstm)
            x_transformer = transformer_input_norm.unnormalize(x_transformer)  # Fixed typo here
            
            # Store results
            results['timestamp'].extend([''.join(str(t) for t in ts) for ts in timestamp.cpu().numpy()])
            results['true_price'].extend(y.cpu().numpy())
            results['estimated_price'].extend(y_hat.cpu().numpy())
            results['strike_price'].extend(x_convlstm[:,0,-1,4].cpu().numpy())
            results['time_to_maturity'].extend(x_convlstm[:,0,-1,2].cpu().numpy())
            results['volume'].extend(x_convlstm[:,0,-1,0].cpu().numpy())
            results['underlying_price'].extend(x_convlstm[:,2,-1,2].cpu().numpy())
            results['cp_flag'].extend(x_convlstm[:,0,-1,1].cpu().numpy())
    
    # Print average metrics
    print(f"\n{dataset_type.upper()} Metrics:")
    avg_loss = sum(metrics_list['loss']) / len(metrics_list['loss'])
    avg_map = sum(metrics_list['map']) / len(metrics_list['map'])
    avg_mape = sum(metrics_list['mape']) / len(metrics_list['mape'])
    avg_corr = sum(metrics_list['corr']) / len(metrics_list['corr'])
    
    print(f"Average Loss: {avg_loss:.5f}")
    print(f"Average MAP: {avg_map:.5f}")
    print(f"Average MAPE: {avg_mape:.5f}")
    print(f"Average Correlation: {avg_corr:.5f}")
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="dualnetwork model prediction script")
    parser.add_argument('-batch_size', type=int, default=64, help="Batch size for prediction")
    parser.add_argument('-session_name', type=str, default=datetime.now().strftime('%b%d_%H%M%S'),
                      help="Session name for saving results")
    parser.add_argument('-test_checkpoint', type=str, default="checkpoints/Feb08_001054_best.pth",
                      help="Path to model checkpoint (.pth file)")
    parser.add_argument('-config', type=str, default=None,
                      help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            args.__dict__.update(config)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize data processor
    data_dir = base_dir / "data/torch-data"
    processor = DataProcessor(data_dir)
    
    # Load and process datasets
    datasets = {}
    normalizations = {}
    for dataset_type in ['test']:   # only for test dataset
        input_tensor_convlstm, input_tensor_transformer, label_tensor, timestamp_tensor, input_tensor_convlstm_norm, input_tensor_transformer_norm, label_norm = processor.load_and_normalize(dataset_type)
        datasets[dataset_type] = get_data_loader(input_tensor_convlstm, input_tensor_transformer, label_tensor, timestamp_tensor, args.batch_size)
        normalizations[dataset_type] = (input_tensor_convlstm_norm, input_tensor_transformer_norm, label_norm)

    # Load model from checkpoint
    model = load_model(args.test_checkpoint, device)
    
    # Generate predictions for each dataset
    for dataset_type, loader in datasets.items():
        input_tensor_convlstm_norm, input_tensor_transformer_norm, label_norm = normalizations[dataset_type]
        results_df = predict_dataset(model, loader, dataset_type, input_tensor_convlstm_norm, input_tensor_transformer_norm, label_norm, device, args)
        
        # Save results
        output_path = Path(f'{Path(args.test_checkpoint).stem}_{dataset_type}_predicted_result.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
