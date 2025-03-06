"""
Calculate various metrics for model evaluation including MSE, RMSE, MAP, MAPE, and correlation.
"""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_metrics(true_values: np.ndarray, pred_values: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics between true and predicted values.
    
    Args:
        true_values: Array of actual values
        pred_values: Array of predicted values
    
    Returns:
        Dictionary containing calculated metrics
    """
    # Vectorized calculations
    mse = np.mean(np.square(true_values - pred_values))
    rmse = np.sqrt(mse)
    map_error = np.mean(np.abs(true_values - pred_values))
    mape = np.mean(np.abs((true_values - pred_values) / true_values)) * 100
    corr = np.corrcoef(true_values, pred_values)[0,1]
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAP': map_error,
        'MAPE': mape,
        'Correlation': corr
    }


def evaluate_predictions(
    file_prefix: str = 'Feb01_113359_convlstm',
    datasets: List[str] = ['train', 'test', 'valid'],
    save_results: bool = False,
    output_file: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model predictions across multiple datasets.
    
    Args:
        file_prefix: Prefix of the CSV files containing predictions
        datasets: List of dataset types to evaluate
        save_results: Whether to save results to file
        output_file: Path to save results if save_results is True
    
    Returns:
        Dictionary containing metrics for each dataset
    """
    results = {}
    
    for dataset in datasets:
        filename = f'{file_prefix}_{dataset}_predicted_result.csv'
        file_path = Path(filename)
        
        try:
            df = pd.read_csv(file_path)
            
            # Extract true and predicted values using pandas vectorized operations
            true_values = df['true price'].values
            pred_values = df['estimated price'].values
            
            # Calculate metrics
            metrics = calculate_metrics(true_values, pred_values)
            results[dataset] = metrics
            
            # Print results
            print(f'\n{dataset.upper()} prediction results:')
            for metric, value in metrics.items():
                print(f"{metric}: {value:.5f}")
                
        except FileNotFoundError:
            print(f"Error: File not found - {filename}")
        except Exception as e:
            print(f"Error processing {dataset} dataset: {str(e)}")
    
    # Save results if requested
    if save_results and output_file:
        try:
            # Convert nested dict to DataFrame for easier saving
            results_df = pd.DataFrame.from_dict(results, orient='index')
            results_df.to_csv(output_file)
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    return results


if __name__ == '__main__':
    # Example usage with default parameters
    results = evaluate_predictions()
