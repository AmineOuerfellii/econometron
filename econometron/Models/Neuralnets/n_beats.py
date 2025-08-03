import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR ,StepLR
from sklearn.model_selection import ParameterGrid
from econometron.utils.data_preparation import  StandardScaler,mean_absolute_error, mean_squared_error,r2_score,root_mean_squared_error,mean_absolute_percentage_error
from typing import Dict, List, Tuple,Union
import time
import warnings

class generic_basis(nn.Module):
    """
    Generic basis function for N-BEATS.
    
    This basis implements a simple linear transformation of the input parameters theta_b and theta_f.
    Used for the 'generic' block type in N-BEATS.
    
    Args:
        backcast_length (int): Length of the backcast window.
        forecast_length (int): Length of the forecast horizon.
    """
    def __init__(self,backcast_length,forecast_length):
      super(generic_basis, self).__init__()
      self.backcast_length = backcast_length
      self.forecast_length = forecast_length
      ###
      #y_Hat_{l}=V_{l}**f * theta_{l}**f
      #x_Hat_{l}=V_{l}**b * theta_{l}**b
      self.basis_b = nn.Parameter(torch.ones(1) * 0.01, requires_grad=True)
      self.basis_f = nn.Parameter(torch.ones(1) * 0.01, requires_grad=True)
      # Bias terms
      self.b_f = nn.Parameter(torch.zeros(forecast_length))
      self.b_b = nn.Parameter(torch.zeros(backcast_length))
      ##
    def forward(self,theta_b,theta_f):
      # print("theta_b shape:", theta_b.shape)
      # print("theta_f shape:", theta_f.shape)
      # print("basis_f shape:", self.basis_f.shape)
      # print("basis_b shape:", self.basis_b.shape)
      backcast=theta_b*self.basis_b+self.b_b
      forecast=theta_f*self.basis_f+self.b_f
      return forecast,backcast
class polynomial_basis(nn.Module):
    """
    Polynomial basis function for N-BEATS.
    
    Constructs polynomial basis matrices for backcast and forecast using powers of normalized time indices.
    Used for interpretable N-BEATS blocks.
    
    Args:
        degree (int): Degree of the polynomial basis.
        backcast_length (int): Length of the backcast window.
        forecast_length (int): Length of the forecast horizon.
    """
    def __init__(self, degree, backcast_length, forecast_length):
        super(polynomial_basis, self).__init__()
        self.degree = degree
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        #####
        #let's define the basis
        # we begin mohaaaa making normalization
        #so the vector of poly is the time index
        T_forecast_prime = np.linspace(0, 1, forecast_length + 1)[:-1]
        T_backcast_prime = np.linspace(0, 1, backcast_length + 1)[:-1]
        basis_forecast = np.zeros((degree + 1, forecast_length))  # Changed: (degree+1, forecast_length)
        basis_backcast = np.zeros((degree + 1, backcast_length))  # Changed: (degree+1, backcast_length)
        for i in range(degree + 1):
            basis_forecast[i, :] = T_forecast_prime ** i
            basis_backcast[i, :] = T_backcast_prime ** i
        self.register_buffer('forecast_basis', torch.tensor(basis_forecast, dtype=torch.float32))
        self.register_buffer('backcast_basis', torch.tensor(basis_backcast, dtype=torch.float32))

    def forward(self, theta_b, theta_f):
        # print("theta_b shape:", theta_b.shape)
        # print("theta_f shape:", theta_f.shape)
        # print("basis_f shape:", self.forecast_basis.shape)
        # print("basis_b shape:", self.backcast_basis.shape)
        forecast = torch.matmul(theta_f, self.forecast_basis)
        backcast = torch.matmul(theta_b, self.backcast_basis)
        return forecast, backcast
class chebyshev_basis(nn.Module):
    """
    Chebyshev polynomial basis for N-BEATS (experimental).
    
    Uses Chebyshev polynomials of the first kind as basis functions for backcast and forecast.
    Intended for improved interpretability and performance in some cases.
    
    Args:
        backcast_length (int): Length of the backcast window.
        forecast_length (int): Length of the forecast horizon.
        degree (int): Degree of the Chebyshev polynomial basis.
    """
    def __init__(self, backcast_length, forecast_length, degree):
        super(chebyshev_basis, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.degree = degree
        t_back = np.linspace(-1, 1, backcast_length)
        t_fore = np.linspace(-1, 1, forecast_length)
        basis_back = np.zeros((degree + 1,backcast_length))
        basis_fore = np.zeros((degree + 1,forecast_length))
        for i in range(backcast_length):
            basis_back[0,i] = 1.0
            if degree >= 1:
                basis_back[1,i] = t_back[i]
            for n in range(2, degree + 1):
                basis_back[n,i] = 2 * t_back[i] * basis_back[n-1,i] - basis_back[n-2,i]
        for i in range(forecast_length):
            basis_fore[0,i] = 1.0
            if degree >= 1:
                basis_fore[1,i] = t_fore[i]
            for n in range(2, degree + 1):
                basis_fore[n,i] = 2 * t_fore[i] * basis_fore[n-1,i] - basis_fore[n-2,i]
        self.register_buffer('forecast_basis', torch.tensor(basis_fore, dtype=torch.float32))
        self.register_buffer('backcast_basis', torch.tensor(basis_back, dtype=torch.float32))
    def forward(self, theta_b,theta_f):
          forecast = torch.matmul(theta_f,self.forecast_basis)
          backcast = torch.matmul(theta_b,self.backcast_basis)
          return forecast,backcast
class fourier_basis(nn.Module):
    """
    Fourier basis function for N-BEATS.
    
    Constructs sine and cosine basis matrices for backcast and forecast, as described in the N-BEATS paper.
    Used for blocks modeling seasonality.
    
    Args:
        backcast_length (int): Length of the backcast window.
        forecast_length (int): Length of the forecast horizon.
    """
    def __init__(self, backcast_length, forecast_length):
        super(fourier_basis, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        # match the reference in the Nbeats paper
        #inchallah in the next update ,i'll give the user the choice for harmonics but now leave that way
        self.H_back = backcast_length // 2 - 1
        self.H_fore = forecast_length // 2 - 1     
        # print("H_back:", self.H_back)
        # print("H_fore:", self.H_fore)
        self.basis_size_back = 2 * self.H_back
        self.basis_size_fore = 2 * self.H_fore
        t_back = np.arange(backcast_length, dtype=np.float32) / backcast_length
        t_fore = np.arange(forecast_length, dtype=np.float32) / forecast_length
        basis_back = np.zeros((self.basis_size_back, backcast_length))
        basis_fore = np.zeros((self.basis_size_fore, forecast_length))
        for l in range(1, self.H_back + 1):
            basis_back[2*(l-1), :] = np.cos(2 * np.pi * l * t_back)
            basis_back[2*(l-1)+1, :] = np.sin(2 * np.pi * l * t_back)
        for l in range(1, self.H_fore + 1):
            basis_fore[2*(l-1), :] = np.cos(2 * np.pi * l * t_fore) 
            basis_fore[2*(l-1)+1, :] = np.sin(2 * np.pi * l * t_fore)    
        self.register_buffer('backcast_basis', torch.FloatTensor(basis_back))
        self.register_buffer('forecast_basis', torch.FloatTensor(basis_fore))
    
    def forward(self, theta_b, theta_f):
        forecast = torch.matmul(theta_f, self.forecast_basis)
        backcast = torch.matmul(theta_b, self.backcast_basis)
        return forecast, backcast
class N_beats_Block(nn.Module):
    """
    N-BEATS Block: Core building block for the N-BEATS model.
    
    Each block consists of a fully connected stack, basis function, and theta parameter generators.
    Supports multiple basis types (generic, fourier, chebyshev, polynomial).
    
    Args:
        input_size (int): Input feature size (usually backcast length).
        Horizon (int): Forecast horizon length.
        backcast (int): Backcast window length.
        degree (int): Degree for polynomial/chebyshev basis (if used).
        n_layers (int): Number of fully connected layers in the block.
        Hidden_size (int): Hidden layer size.
        basis_type (str): Basis type ('generic', 'fourier', 'chebyshev', 'polynomial').
    """
    def __init__(self,
                 input_size: int,
                 Horizon: int,
                 backcast: int,
                 degree: int,
                 n_layers: int,
                 Hidden_size: int = 512,
                 basis_type: str = "generic"):
        super(N_beats_Block, self).__init__()
        self.basis_type = basis_type
        self.input_size = input_size
        self.degree = degree
        self.Horizon = Horizon
        self.backcast = backcast
        #####
        # we will set 4 layers for teh Fully connected stack
        # # # #  self.FC_stack=nn.Sequential(
        # # # #      nn.Linear(in_features=input_size,out_features=Hidden_size),
        # # # #      nn.ReLU(),#h(l1)
        # # # #      nn.Linear(in_features=Hidden_size,out_features=Hidden_size),
        self.FC_stack = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=Hidden_size),
            nn.ReLU(),
            *[
            layer
            for _ in range(n_layers - 1)
            for layer in (nn.Linear(in_features=Hidden_size, out_features=Hidden_size), nn.ReLU())
            ]
        )
        # # # #      nn.Linear(in_features=Hidden_size,out_features=Hidden_size),
        # # # #      nn.ReLU(),#h(l3)
        # # # #      nn.Linear(in_features=Hidden_size,out_features=Hidden_size),
        # # # #      nn.ReLU() #h(l4)
        # # # #  )
        # layers = []
        # for i in range(n_layers):
        #     layers.append(nn.Linear(in_features=input_size, out_features=Hidden_size))
        #     layers.append(nn.ReLU())
        #     input_size = Hidden_size
        # self.FC_stack = nn.Sequential(*layers)
        # see page 3 in N-BEATS paper by Boris N. Oreshkin
        # Now we prepare for the FC layer within each basis type choice
        # please contact me @mohamedamine.ouerfelli@outlook.com specially
        # if the matter concerns the architecture of the model and the single FC layer in the Nbeats block with generates theta
        if basis_type in ['generic', 'fourier', 'chebyshev', 'polynomial']:
            self.basis = basis_type
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")
        if self.basis == 'generic':
            self.theta_f = nn.Linear(in_features=Hidden_size, out_features=Horizon)
            self.theta_b = nn.Linear(in_features=Hidden_size, out_features=backcast)
            self.basis_function = generic_basis(backcast_length=backcast, forecast_length=Horizon)
        elif self.basis == 'fourier':
            theta_Hor = self.Horizon // 2 - 1
            theta_back = self.backcast // 2 - 1
            self.theta_f = nn.Linear(in_features=Hidden_size, out_features=2 * theta_Hor)
            self.theta_b = nn.Linear(in_features=Hidden_size, out_features=2 * theta_back)
            self.basis_function = fourier_basis(backcast_length=backcast, forecast_length=Horizon)
        elif self.basis == 'chebyshev':
            if degree is None:
                degree = 3
            self.theta_f = nn.Linear(in_features=Hidden_size, out_features=self.degree + 1)
            self.theta_b = nn.Linear(in_features=Hidden_size, out_features=self.degree + 1)
            self.basis_function = chebyshev_basis(backcast_length=backcast, forecast_length=Horizon, degree=degree)
        elif self.basis == 'polynomial':
            if degree is None:
                degree = 3
            self.theta_f = nn.Linear(in_features=Hidden_size, out_features=self.degree + 1)
            self.theta_b = nn.Linear(in_features=Hidden_size, out_features=self.degree + 1)
            self.basis_function = polynomial_basis(backcast_length=backcast, forecast_length=Horizon, degree=degree)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        print("Input shape:", x.shape)
        h_4 = self.FC_stack(x)
        theta_b = self.theta_b(h_4)
        theta_f = self.theta_f(h_4)
        forecast, backcast = self.basis_function(theta_b, theta_f)
        return forecast, backcast
class N_beats_stack(nn.Module):
    """
    N-BEATS Stack: Sequence of N-BEATS blocks.
    
    Each stack contains multiple blocks (optionally sharing weights) and produces a forecast and residual.
    
    Args:
        input_size (int): Input feature size (usually backcast length).
        n_blocks (int): Number of blocks in the stack.
        horizon (int): Forecast horizon length.
        backcast_size (int): Backcast window length.
        n_layers_per_block (int): Number of layers per block.
        hidden_size (int): Hidden layer size.
        degree (int): Degree for polynomial/chebyshev basis (if used).
        basis_type (str): Basis type for all blocks in the stack.
        share_weights (bool): Whether to share weights across blocks.
    """
    def __init__(self, input_size, n_blocks, horizon, backcast_size,n_layers_per_block, hidden_size,
                 degree, basis_type="generic", share_weights=True):
        super(N_beats_stack, self).__init__()
        self.n_blocks = n_blocks
        self.horizon = horizon
        self.backcast_size = backcast_size
        self.share_weights = share_weights
        if share_weights and n_blocks > 0:
            self.shared_block = N_beats_Block(
                input_size=input_size,
                Horizon=self.horizon, 
                backcast=self.backcast_size,
                degree=degree,
                n_layers=n_layers_per_block,
                Hidden_size=hidden_size,
                basis_type=basis_type
            )
            self.blocks = nn.ModuleList([self.shared_block for _ in range(n_blocks)])
        else:
            self.blocks = nn.ModuleList([
                N_beats_Block(
                    input_size=input_size,
                    Horizon=self.horizon, 
                    backcast=self.backcast_size, 
                    degree=degree,
                    n_layers=n_layers_per_block,
                    Hidden_size=hidden_size,
                    basis_type=basis_type,
                ) for _ in range(n_blocks)
            ])
    def forward(self, x):
        if self.n_blocks == 0:
            raise ValueError("Number of blocks must be greater than 0")
        residual = x
        stack_forecast = torch.zeros(x.shape[0], self.horizon, device=x.device)
        # print(stack_forecast.shape)
        for block in self.blocks:
            forecast, backcast = block(residual)
            stack_forecast += forecast
            residual = residual - backcast    
        return stack_forecast, residual
class N_beats(nn.Module):
    """
    N-BEATS Model: Main model class for N-BEATS time series forecasting.
    
    Composed of multiple stacks, each with its own configuration. Handles forward pass and model info.
    
    Args:
        stack_configs (List[Dict]): List of stack configuration dictionaries.
        backcast_length (int): Length of the input (backcast) window.
        forecast_length (int): Length of the forecast horizon.
    """
    def __init__(self, stack_configs, backcast_length, forecast_length):
        super(N_beats, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stack_configs = stack_configs
        self.stacks = nn.ModuleList()       
        for config in stack_configs:
            n_blocks = config['n_blocks']
            basis_type = config['basis_type']
            n_layers_per_block = config.get('n_layers_per_block', 4)
            hidden_size = config.get('hidden_size', 512)
            degree = config.get('degree', 3)
            share_weights = config.get('share_weights', True)    
            stack = N_beats_stack(
                input_size=backcast_length,
                n_blocks=n_blocks,
                horizon=forecast_length,
                backcast_size=backcast_length,
                n_layers_per_block=n_layers_per_block,
                hidden_size=hidden_size,
                degree=degree,
                basis_type=basis_type,
                share_weights=share_weights)          
            self.stacks.append(stack)    
    def forward(self, x):
        residual = x
        total_forecast = torch.zeros(x.shape[0], self.forecast_length, device=x.device)
        for stack in self.stacks:
            stack_forecast, residual = stack(residual)
            print("stack_forecast shape:", stack_forecast.shape)
            print("residual shape:", residual.shape)
            total_forecast += stack_forecast           
        return total_forecast
    def get_model_info(self):
        info = {
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
            'num_stacks': len(self.stacks),
            'stack_configs': []
        }     
        for i, config in enumerate(self.stack_configs):
            stack_info = {
                'stack_id': i,
                'n_blocks': config['n_blocks'],
                'basis_type': config['basis_type'],
                'hidden_size': config.get('hidden_size', 512),
                'n_layers_per_block': config.get('n_layers_per_block', 4)
            }
            if config['basis_type'] in ['polynomial', 'chebyshev']:
                stack_info['degree'] = config.get('degree', 3)
            info['stack_configs'].append(stack_info)
        return info
class LossCalculator:
    """
    Utility class for various loss functions used in N-BEATS training.
    
    Provides static methods for MSE, MAE, MAPE, SMAPE, and Huber loss.
    """
    
    @staticmethod
    def mse_loss(y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mae_loss(y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))
    
    @staticmethod
    def mape_loss(y_true, y_pred, epsilon=1e-8):
        return torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon))) * 100
    
    @staticmethod
    def smape_loss(y_true, y_pred, epsilon=1e-8):
        numerator = torch.abs(y_true - y_pred)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + epsilon
        return torch.mean(numerator / denominator) * 100
    
    @staticmethod
    def huber_loss(y_true, y_pred, delta=1.0):
        residual = torch.abs(y_true - y_pred)
        condition = residual < delta
        squared_loss = 0.5 * residual ** 2
        linear_loss = delta * residual - 0.5 * delta ** 2
        return torch.mean(torch.where(condition, squared_loss, linear_loss))
class EarlyStopping:
    """
    Early stopping utility class for N-BEATS training.
    
    Monitors validation loss and stops training if no improvement is seen for a given patience.
    Optionally restores the best model weights.
    
    Args:
        patience (int): Number of epochs to wait for improvement.
        min_delta (float): Minimum change to qualify as improvement.
        restore_best_weights (bool): Whether to restore best weights after stopping.
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()
class NeuralForecast:
    """
    Complete forecasting framework for N-BEATS model.
    
    Provides data processing, training, evaluation, hyperparameter search, and plotting utilities for N-BEATS.
    Handles normalization, device management, and model saving/loading.
    
    Args:
        stack_configs (List[Dict]): List of stack configuration dictionaries.
        backcast_length (int): Length of the input (backcast) window.
        forecast_length (int): Length of the forecast horizon.
        device (str, optional): Device to use ('cpu', 'cuda', or None for auto-detection).
    """
    
    def __init__(self, stack_configs: List[Dict], backcast_length: int, forecast_length: int, device: str = None):
        """
        Initialize NeuralForecast
        
        Args:
            stack_configs: List of stack configurations for N-BEATS
            backcast_length: Length of input sequence
            forecast_length: Length of forecast horizon
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
        """
        self.stack_configs = stack_configs
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = N_beats(stack_configs, backcast_length, forecast_length).to(self.device)
        
        # Training attributes
        self.history = {}
        self.scaler = None
        self.is_fitted = False
        
        print(f"NeuralForecast initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def process_data(self, data: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                     normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process time series data into train/val/test splits
        
        Args:
            data: Input time series data
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data  
            normalize: Whether to normalize the data
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if normalize:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._create_sequences(data, self.backcast_length, self.forecast_length)
        
        # Split data
        total_samples = len(X)
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        # Convert to tensors and combine X,y
        train_data = torch.FloatTensor(np.concatenate([X_train, y_train], axis=1))
        val_data = torch.FloatTensor(np.concatenate([X_val, y_val], axis=1))
        test_data = torch.FloatTensor(np.concatenate([X_test, y_test], axis=1))
        
        # Squeeze last dimension if data is 3D with last dim == 1
        if train_data.ndim == 3 and train_data.shape[-1] == 1:
            train_data = train_data.squeeze(-1)
        if val_data.ndim == 3 and val_data.shape[-1] == 1:
            val_data = val_data.squeeze(-1)
        if test_data.ndim == 3 and test_data.shape[-1] == 1:
            test_data = test_data.squeeze(-1)
        print(f"Data processed - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _create_sequences(self, data: np.ndarray, backcast_length: int, forecast_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences from time series data"""
        X, y = [], []
        for i in range(len(data) - backcast_length - forecast_length + 1):
            X.append(data[i:i+backcast_length])
            y.append(data[i+backcast_length:i+backcast_length+forecast_length])
        return np.array(X), np.array(y)
    
    def fit(self, train_data: torch.Tensor, val_data: torch.Tensor = None, epochs: int = 100,
            batch_size: int = 32, learning_rate: float = 1e-3, optimizer: str = 'adam',
            loss_function: str = 'mse', early_stopping: bool = False, patience: int = 10,
            scheduler: str = None, gradient_clip: float = None, verbose: bool = True,
            normalize: bool = False) -> Dict:
        """
        Train the N-BEATS model
        
        Args:
            train_data: Training data tensor (concatenated X and y)
            val_data: Validation data tensor  
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            loss_function: Loss function ('mse', 'mae', 'huber', 'mape', 'smape')
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            scheduler: Learning rate scheduler ('step', 'cosine', 'plateau')
            gradient_clip: Gradient clipping value
            verbose: Whether to print training progress
            normalize: Whether to normalize data
            
        Returns:
            Training history dictionary
        """
        print("Starting training...")
        
        # Setup data normalization
        if normalize and self.scaler is None:
            train_X = train_data[:, :self.backcast_length]
            self.scaler = StandardScaler()
            self.scaler.fit(train_X.numpy())
        
        # Setup data loaders
        train_loader = self._create_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_data, batch_size, shuffle=False) if val_data is not None else None
        
        # Setup optimizer
        if optimizer.lower() == 'adam':
            opt = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer.lower() == 'adamw':
            opt = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        elif optimizer.lower() == 'sgd':
            opt = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Setup loss function
        loss_fn = self._get_loss_function(loss_function)
        
        # Setup scheduler
        scheduler_obj = None
        if scheduler:
            if scheduler.lower() == 'step':
                scheduler_obj = StepLR(opt, step_size=30, gamma=0.5)
            elif scheduler.lower() == 'cosine':
                scheduler_obj = CosineAnnealingLR(opt, T_max=epochs)
            elif scheduler.lower() == 'plateau':
                scheduler_obj = ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
        
        # Setup early stopping
        early_stopping_obj = None
        if early_stopping:
            early_stopping_obj = EarlyStopping(patience=patience)
        
        # Training loop
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader, opt, loss_fn, gradient_clip, normalize)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader, loss_fn, normalize)
                self.history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            # Learning rate tracking
            current_lr = opt.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Scheduler step
            if scheduler_obj:
                if isinstance(scheduler_obj, ReduceLROnPlateau):
                    scheduler_obj.step(val_loss if val_loss else train_loss)
                else:
                    scheduler_obj.step()
            
            # Early stopping check
            if early_stopping_obj and val_loss:
                if early_stopping_obj(val_loss, self.model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Progress logging
            if verbose and (epoch + 1) % 10 == 0:
                epoch_time = time.time() - start_time
                val_msg = f", Val Loss: {val_loss:.6f}" if val_loss else ""
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}{val_msg}, "
                      f"LR: {current_lr:.2e}, Time: {epoch_time:.2f}s")
        
        self.is_fitted = True
        print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
        
        return self.history
    
    def _create_dataloader(self, data: torch.Tensor, batch_size: int, shuffle: bool = False) -> DataLoader:
        """Create DataLoader from data tensor"""
        X = data[:, :self.backcast_length]
        y = data[:, self.backcast_length:]
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _get_loss_function(self, loss_type: str):
        """Get loss function by name"""
        if loss_type.lower() == 'mse':
            return LossCalculator.mse_loss
        elif loss_type.lower() == 'mae':
            return LossCalculator.mae_loss
        elif loss_type.lower() == 'huber':
            return LossCalculator.huber_loss
        elif loss_type.lower() == 'mape':
            return LossCalculator.mape_loss
        elif loss_type.lower() == 'smape':
            return LossCalculator.smape_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")
    
    def _train_epoch(self, train_loader: DataLoader, optimizer, loss_fn, gradient_clip: float = None, 
                     normalize: bool = False) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            if normalize and self.scaler:
                batch_X = torch.tensor(self.scaler.transform(batch_X.cpu().numpy()), 
                                     dtype=torch.float32, device=self.device)
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            # If outputs is a tuple (forecast, backcast), use only forecast for loss
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = loss_fn(batch_y, outputs)
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader, loss_fn, normalize: bool = False) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                if normalize and self.scaler:
                    batch_X = torch.tensor(self.scaler.transform(batch_X.cpu().numpy()), 
                                         dtype=torch.float32, device=self.device)
                outputs = self.model(batch_X)
                # If outputs is a tuple (forecast, backcast), use only forecast for loss
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = loss_fn(batch_y, outputs)
                total_loss += loss.item()
        return total_loss / len(val_loader)
    
    def plot_training_history(self, figsize: Tuple[int, int] = (15, 5)):
        """Plot training history"""
        if not self.history:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot losses
        axes[0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Val Loss', color='red')
        axes[0].set_title('Training History')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot learning rate
        axes[1].plot(self.history['learning_rate'], color='green')
        axes[1].set_title('Learning Rate')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_yscale('log')
        axes[1].grid(True)
        
        # Plot loss ratio (if validation exists)
        if self.history['val_loss']:
            val_train_ratio = [v/t for v, t in zip(self.history['val_loss'], self.history['train_loss'])]
            axes[2].plot(val_train_ratio, color='purple')
            axes[2].set_title('Val/Train Loss Ratio')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Ratio')
            axes[2].grid(True)
        else:
            axes[2].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Validation Info')
        
        plt.tight_layout()
        plt.show()
    
    def forecast(self, input_sequence: Union[np.ndarray, torch.Tensor], return_components: bool = False,
                 plot_forecast: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Generate forecast for input sequence
        
        Args:
            input_sequence: Input sequence of length backcast_length
            return_components: Whether to return stack components
            plot_forecast: Whether to plot the forecast
            
        Returns:
            Forecast array or tuple of (forecast, components)
        """
        if not self.is_fitted:
            warnings.warn("Model is not fitted. Please train the model first.")
        
        self.model.eval()
        
        # Prepare input
        if isinstance(input_sequence, np.ndarray):
            input_sequence = torch.FloatTensor(input_sequence)
        
        if len(input_sequence.shape) == 1:
            input_sequence = input_sequence.unsqueeze(0)
        
        input_sequence = input_sequence.to(self.device)
        
        # Normalize if needed
        if self.scaler:
            input_np = input_sequence.cpu().numpy()
            input_scaled = self.scaler.transform(input_np)
            input_sequence = torch.tensor(input_scaled, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            if return_components:
                # Get individual stack contributions
                components = {}
                residual = input_sequence
                total_forecast = torch.zeros(input_sequence.shape[0], self.forecast_length, device=self.device)
                
                for i, stack in enumerate(self.model.stacks):
                    stack_forecast, residual = stack(residual)
                    total_forecast += stack_forecast
                    
                    # Store component
                    component_name = f"stack_{i}_{self.stack_configs[i]['basis_type']}"
                    components[component_name] = stack_forecast.cpu().numpy()
                
                forecast = total_forecast.cpu().numpy()
            else:
                forecast = self.model(input_sequence).cpu().numpy()
                components = None
        
        # Denormalize if needed
        if self.scaler:
            forecast_reshaped = forecast.reshape(-1, 1)
            forecast_denorm = self.scaler.inverse_transform(forecast_reshaped)
            forecast = forecast_denorm.reshape(forecast.shape)
            
            if components:
                for key, comp in components.items():
                    comp_reshaped = comp.reshape(-1, 1)
                    comp_denorm = self.scaler.inverse_transform(comp_reshaped)
                    components[key] = comp_denorm.reshape(comp.shape)
        
        if plot_forecast:
            self._plot_single_forecast(input_sequence.cpu().numpy(), forecast, components)
        
        if return_components:
            return forecast, components
        return forecast
    
    def plot_forecast(self, historical_data: np.ndarray, forecast_data: np.ndarray = None, 
                      plot_components: bool = False, figsize: Tuple[int, int] = (15, 8)):
        """
        Plot historical data with forecast
        
        Args:
            historical_data: Historical time series data
            forecast_data: Actual future values for comparison (optional)
            plot_components: Whether to plot stack components
            figsize: Figure size
        """
        if not self.is_fitted:
            print("Model is not fitted. Cannot generate forecast.")
            return
        
        # Get input sequence (last part of historical data)
        input_sequence = historical_data[-self.backcast_length:]
        
        # Generate forecast
        forecast, components = self.forecast(input_sequence, return_components=True)
        forecast = forecast.flatten()
        
        # Create time indices
        hist_time = np.arange(len(historical_data))
        forecast_time = np.arange(len(historical_data), len(historical_data) + len(forecast))
        
        if plot_components and components:
            # Create subplots for components
            n_components = len(components)
            fig, axes = plt.subplots(n_components + 1, 1, figsize=(figsize[0], figsize[1] * (n_components + 1)))
            
            if n_components == 1:
                axes = [axes]
            
            # Main forecast plot
            axes[0].plot(hist_time, historical_data, label='Historical', color='blue', alpha=0.7)
            axes[0].plot(forecast_time, forecast, label='Forecast', color='red', linewidth=2)
            
            if forecast_data is not None:
                axes[0].plot(forecast_time[:len(forecast_data)], forecast_data, 
                           label='Actual', color='green', linewidth=2, linestyle='--')
            
            axes[0].axvline(x=len(historical_data)-1, color='black', linestyle=':', alpha=0.7)
            axes[0].set_title('N-BEATS Forecast')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Component plots
            for i, (name, component) in enumerate(components.items()):
                axes[i+1].plot(forecast_time, component.flatten(), label=f'{name}', linewidth=2)
                axes[i+1].set_title(f'Component: {name}')
                axes[i+1].legend()
                axes[i+1].grid(True, alpha=0.3)
                
        else:
            # Simple forecast plot
            plt.figure(figsize=figsize)
            plt.plot(hist_time, historical_data, label='Historical', color='blue', alpha=0.7)
            plt.plot(forecast_time, forecast, label='Forecast', color='red', linewidth=2)
            
            if forecast_data is not None:
                plt.plot(forecast_time[:len(forecast_data)], forecast_data, 
                        label='Actual', color='green', linewidth=2, linestyle='--')
            
            plt.axvline(x=len(historical_data)-1, color='black', linestyle=':', alpha=0.7)
            plt.title('N-BEATS Forecast')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_single_forecast(self, input_sequence: np.ndarray, forecast: np.ndarray, 
                             components: Dict = None):
        """Plot a single forecast"""
        input_flat = input_sequence.flatten()
        forecast_flat = forecast.flatten()
        
        # Time indices
        input_time = np.arange(len(input_flat))
        forecast_time = np.arange(len(input_flat), len(input_flat) + len(forecast_flat))
        
        plt.figure(figsize=(12, 6))
        plt.plot(input_time, input_flat, label='Input Sequence', color='blue')
        plt.plot(forecast_time, forecast_flat, label='Forecast', color='red', linewidth=2)
        plt.axvline(x=len(input_flat)-1, color='black', linestyle=':', alpha=0.7)
        plt.title('Forecast from Input Sequence')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def evaluate(self, test_data: torch.Tensor, metrics: List[str] = ['mae', 'mse', 'rmse']) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data tensor
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            print("Model is not fitted. Please train the model first.")
            return {}
        
        self.model.eval()
        
        # Prepare data
        X_test = test_data[:, :self.backcast_length]
        y_test = test_data[:, self.backcast_length:]
        
        # Generate predictions
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for i in range(len(X_test)):
                input_seq = X_test[i:i+1].to(self.device)
                
                if self.scaler:
                    input_np = input_seq.cpu().numpy()
                    input_scaled = self.scaler.transform(input_np)
                    input_seq = torch.tensor(input_scaled, dtype=torch.float32, device=self.device)
                
                pred = self.model(input_seq).cpu().numpy()
                
                if self.scaler:
                    pred_reshaped = pred.reshape(-1, 1)
                    pred_denorm = self.scaler.inverse_transform(pred_reshaped)
                    pred = pred_denorm.reshape(pred.shape)
                
                predictions.append(pred.flatten())
                actuals.append(y_test[i].numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        results = {}
        
        for metric in metrics:
            if metric.lower() == 'mae':
                results['MAE'] = mean_absolute_error(actuals, predictions)
            elif metric.lower() == 'mse':
                results['MSE'] = mean_squared_error(actuals, predictions)
            elif metric.lower() == 'rmse':
                results['RMSE'] = root_mean_squared_error(actuals, predictions)
            elif metric.lower() == 'mape':
                results['MAPE'] = mean_absolute_percentage_error(actuals, predictions)
            elif metric.lower() == 'smape':
                # SMAPE calculation
                numerator = np.abs(actuals - predictions)
                denominator = (np.abs(actuals) + np.abs(predictions)) / 2
                results['SMAPE'] = np.mean(numerator / denominator) * 100
            elif metric.lower() == 'r2':
                results['R2'] = r2_score(actuals, predictions)
        
        # Print results
        print("Evaluation Results:")
        print("-" * 30)
        for metric, value in results.items():
            print(f"{metric}: {value:.6f}")
        
        return results
    
    def hyperparameter_finder(self, train_data: torch.Tensor, val_data: torch.Tensor,
                             param_grid: Dict, max_trials: int = 10, epochs: int = 50) -> Dict:
        """
        Basic hyperparameter search
        
        Args:
            train_data: Training data
            val_data: Validation data
            param_grid: Dictionary of parameters to search
            max_trials: Maximum number of trials
            epochs: Epochs per trial
            
        Returns:
            Best parameters and results
        """
        print(f"Starting hyperparameter search with {max_trials} trials...")
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        
        if len(param_combinations) > max_trials:
            # Random sample if too many combinations
            import random
            param_combinations = random.sample(param_combinations, max_trials)
        
        best_score = float('inf')
        best_params = None
        results = []
        
        for i, params in enumerate(param_combinations):
            print(f"\nTrial {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Create new model instance
                temp_model = NeuralForecast(
                    stack_configs=self.stack_configs,
                    backcast_length=self.backcast_length,
                    forecast_length=self.forecast_length,
                    device=self.device
                )
                
                # Train with current parameters
                history = temp_model.fit(
                    train_data=train_data,
                    val_data=val_data,
                    epochs=epochs,
                    verbose=False,
                    **params
                )
                
                # Get validation score
                val_score = min(history['val_loss']) if history['val_loss'] else min(history['train_loss'])
                
                results.append({
                    'params': params,
                    'val_score': val_score,
                    'history': history
                })
                
                if val_score < best_score:
                    best_score = val_score
                    best_params = params
                
                print(f"Val Score: {val_score:.6f}")
                
            except Exception as e:
                print(f"Trial failed: {e}")
                continue
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best validation score: {best_score:.6f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'stack_configs': self.stack_configs,
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
            'scaler': self.scaler,
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint.get('scaler')
        self.history = checkpoint.get('history', {})
        self.is_fitted = True
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict:
        """Get model summary information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            'model_info': self.model.get_model_info(),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'is_fitted': self.is_fitted,
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length
        }
        
        return summary