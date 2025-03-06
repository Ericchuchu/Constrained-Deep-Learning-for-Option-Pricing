# 3D Tensor-based Deep Learning Models for Predicting Option Price

This repository contains the implementation of advanced deep learning models for option pricing prediction, as described in the manuscript "3D Tensor-based Deep Learning Models for Predicting Option Price".

## Project Overview

This project explores novel deep learning approaches for option pricing, focusing on 3D tensor-based models that can effectively capture the complex relationships between various market factors. The models are designed to predict option prices based on historical data, outperforming traditional pricing methods in terms of accuracy and adaptability to market conditions.

### Key Features

- Multiple state-of-the-art deep learning architectures for option pricing
- 3D tensor-based data representation for capturing complex market dynamics
- Weighted loss functions based on Kernel Density Estimation (KDE) to handle imbalanced data
- Rolling window training methodology for time series forecasting
- Comprehensive evaluation metrics and visualization tools
- Support for both standard and rolling window training approaches

## Models

The project implements several deep learning architectures:

### MultiPatchFormer (MPF)

A transformer-based model that uses multi-scale patch embedding to process option data. The model is designed to handle 3D tensor input with shape (batch_size, channels, sequence_length, features). Key components include:

- **Multi-Scale Embedding**: Processes input data at different scales using convolutional layers with various kernel sizes and strides to capture patterns at different resolutions
- **Temporal Encoder**: Captures temporal dependencies using transformer encoder layers that model relationships between tokens in the sequence
- **Channel-wise Encoder**: Processes relationships across different features using multi-head attention to capture cross-feature interactions
- **Token Attention**: Combines information from different tokens using an attention mechanism that weights the importance of each token
- **Single-Step Decoder**: Generates the final option price prediction by fusing information across features

The MPF model is implemented in both standard and rolling window versions, allowing for different training methodologies.

### Dual Branch Network

A hybrid architecture that combines ConvLSTM and Transformer branches to leverage the strengths of both approaches:

- **ConvLSTM Branch**: Captures spatio-temporal patterns in the data using convolutional LSTM cells that process the 3D tensor input (batch_size, channels, sequence_length, features)
- **Transformer Branch**: Models long-range dependencies and complex relationships using a transformer encoder with positional encoding for both sequence positions and date information
- **PDE-Guided Loss**: Incorporates financial domain knowledge through partial differential equations based on the Black-Scholes model, calculating derivatives with respect to time and underlying price
- **Fusion Layer**: Combines outputs from both branches using fully connected layers to generate the final prediction

The dual branch approach allows the model to capture both local patterns through the ConvLSTM and global dependencies through the transformer.

### ConvLSTM and CNN-RNN Models

Additional architectures that serve as baselines and alternatives:

- **ConvLSTM**: Combines convolutional and LSTM layers for spatio-temporal modeling
  - Uses ConvLSTM cells with 1D convolutions to process sequential data
  - Includes layer normalization and batch normalization for stable training
  - Implements a multi-layer architecture with skip connections
  - Uses softplus activation for ensuring positive option price outputs

- **CNN-RNN**: Sequential combination of CNN for feature extraction and RNN for temporal modeling
  - Uses multi-scale CNN with different dilation rates to capture patterns at various time scales
  - Implements bidirectional GRU layers for sequence modeling
  - Combines CNN and RNN outputs through residual connections
  - Uses group normalization for improved training stability

## Data Processing

The project includes comprehensive data processing pipelines:

- **Data Preprocessing**: Cleans and transforms raw option data
- **Feature Engineering**: Calculates financial metrics like implied volatility, Greeks, and moneyness
- **Normalization**: Standardizes input features for better model training
- **Tensor Creation**: Converts processed data into 3D tensors for model input

## Training Methodologies

### Standard Training

- Splits data into training, validation, and test sets based on time periods
- Uses weighted loss based on Kernel Density Estimation (KDE) to handle data imbalance:
  ```python
  # Extract moneyness values
  moneyness = x_input[:, 0, -1, 4].detach().cpu().numpy()
  # Calculate weights using KDE
  weights = 1.0 / (kde_model(moneyness.reshape(1, -1)) + 1e-6)
  # Normalize weights
  weights = weights / mean_weights
  # Compute weighted MSE loss
  mse_loss = torch.mean(weights * (predicted - target) ** 2)
  ```
- Implements early stopping with patience to prevent overfitting
- Uses learning rate scheduling with ReduceLROnPlateau to adapt learning rates
- Applies gradient clipping to prevent exploding gradients
- Tracks and visualizes training progress with detailed loss curves

### Rolling Window Training

- Implements a time-based rolling window approach for time series forecasting:
  ```
  Window 1: Train[t₁:t₂] → Valid[t₂:t₃] → Test[t₃:t₄]
  Window 2: Train[t₂:t₅] → Valid[t₅:t₆] → Test[t₆:t₇]
  ...
  ```
- Supports both fixed-window and expanding-window methodologies
- Enables transfer learning between consecutive windows by initializing each window's model with the previous window's weights
- Provides ensemble prediction capabilities by combining predictions from multiple models:
  ```python
  # Adaptive ensemble based on validation performance
  selected_models, weights = adaptive_ensemble(all_window_models, window_data, device)
  # Weighted ensemble prediction
  output = ensemble_predict(selected_models, weights, x_input, device)
  ```
- Visualizes performance metrics across windows to analyze temporal patterns

## Project Structure

### Data Folder
- Contains option and stock data
- Includes raw CSV files, processed data, and PyTorch tensors
- Key files:
  - `prs_dataset_mpf.csv`, `prs_dataset_dual.csv`: Processed datasets
  - `torch-data/`: Contains preprocessed PyTorch tensors for training

### Core Folder
- Contains model implementations and data processing scripts
- Key files:
  - `models_multi_patch_former.py`, `models_multi_patch_former_adjusted.py`: MPF model implementations
  - `models_dual_network.py`: Dual branch network implementation
  - `models_convlstm.py`: ConvLSTM model implementation
  - `models_CNN_RNN.py`: CNN-RNN model implementation
  - `data_preprocess_*.py`: Data preprocessing pipelines
  - `functions.py`: Utility functions for option pricing and metrics

### Main Folder
- Contains training and evaluation scripts
- Key files:
  - `main_mpf_network.py`: Standard MPF training script
  - `main_mpf_rolling_network.py`: Rolling window MPF training script
  - `main_dual_network.py`: Dual network training script
  - `main_convlstm.py`: ConvLSTM training script
  - `main_CNN_RNN.py`: CNN-RNN training script
  - `visualization.ipynb`: Result visualization notebook
  - `checkpoints/`: Saved model weights and optimizer states

## Usage

### Data Preparation

1. Preprocess raw option data:
   ```
   python core/data_preprocess_taiex_option_type2_mpf.py
   ```

2. For dual network data:
   ```
   python core/data_preprocess_taiex_option_type2_dual.py
   ```

### Model Training

1. Train the MultiPatchFormer model:
   ```
   python main/main_mpf_network.py -max_epoch 100 -batch_size 64 -early_stop_mode True
   ```

2. Train with rolling window approach:
   ```
   python main/main_mpf_rolling_network.py -max_epoch 50 -batch_size 64 -early_stop_mode True
   ```

3. Train the Dual Branch Network:
   ```
   python main/main_dual_network.py -max_epoch 100 -batch_size 64 -early_stop_mode True
   ```

### Evaluation and Visualization

1. Test a trained model:
   ```
   python main/main_mpf_network.py -test_checkpoint checkpoints/mpf_network_best.pth
   ```

2. Visualize results:
   ```
   jupyter notebook main/visualization.ipynb
   ```

## Results

The models are evaluated using multiple metrics:

- Mean Absolute Percentage Error (MAPE): Measures the percentage difference between predicted and actual prices
- Mean Absolute Error (MAE): Measures the absolute difference between predicted and actual prices
- Correlation coefficient: Measures the linear relationship between predicted and actual prices
- MSE loss: Measures the squared difference between predicted and actual prices

Visualization tools include:
- 3D plots of predicted vs. actual option prices with moneyness and time-to-maturity as axes
- Loss curves over training epochs showing training, validation, and test losses
- Time-based performance analysis across different market conditions
- Cumulative performance metrics for rolling window models

Example visualization:
```python
# Create 3D visualization
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(moneyness, time_to_maturity, true_prices, c='blue', marker='o', label='True Prices')
ax.scatter(moneyness, time_to_maturity, estimated_prices, c='red', marker='^', label='Estimated Prices')
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Pandas
- Matplotlib
- SciPy
- tqdm

## Citation

If you use this code in your research, please cite:
```
@article{3DTensorOptionPricing,
  title={3D Tensor-based Deep Learning Models for Predicting Option Price},
  author={},
  journal={},
  year={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
