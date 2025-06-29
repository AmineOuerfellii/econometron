import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import time
from typing import Dict, List, Tuple, Optional, Union
import warnings
from utils.data_preparation.scaler import StandardScaler, MinMaxScaler,mean_absolute_error, mean_squared_error

class generic_basis(nn.Module):
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
      this is experimental thing , I "the author of this package",
      since for interprebality we use a polyniomial basis , i thought we could use the cheb basis , since it more
      performant.
      """
      def __init__(self,
                   backcast_length,
                   forecast_length,
                   degree):
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
        #  self.FC_stack=nn.Sequential(
        #      nn.Linear(in_features=input_size,out_features=Hidden_size),
        #      nn.ReLU(),#h(l1)
        #      nn.Linear(in_features=Hidden_size,out_features=Hidden_size),
        #      nn.ReLU(),#h(l2)
        #      nn.Linear(in_features=Hidden_size,out_features=Hidden_size),
        #      nn.ReLU(),#h(l3)
        #      nn.Linear(in_features=Hidden_size,out_features=Hidden_size),
        #      nn.ReLU() #h(l4)
        #  )
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_features=input_size, out_features=Hidden_size))
            layers.append(nn.ReLU())
            input_size = Hidden_size
        self.FC_stack = nn.Sequential(*layers)
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
        h_4 = self.FC_stack(x)
        theta_b = self.theta_b(h_4)
        theta_f = self.theta_f(h_4)
        forecast, backcast = self.basis_function(theta_b, theta_f)
        return forecast, backcast
class N_beats_stack(nn.Module):
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
        return stack_forecast, backcast
class N_beats(nn.Module):
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
            # print("stack_forecast shape:", stack_forecast.shape)
            # print("residual shape:", residual.shape)
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
class EarlyStopping:
    """Early stopping utility class"""
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

class LossCalculator:
    """Utility class for various loss functions"""
    
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

class NeuralForecast:
    """
    N-BEATS wrapper for time series forecasting
    
    Parameters:
    -----------
    stack_configs : list
        Configuration for each stack in the model
    backcast_length : int
        Length of input sequence
    forecast_length : int
        Length of forecast horizon
    device : str, optional
        Device to run the model on ('cpu' or 'cuda')
    """
    
    def __init__(self, 
                 stack_configs: List[Dict], 
                 backcast_length: int, 
                 forecast_length: int,
                 device: str = None):
        
        self.stack_configs = stack_configs
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize model
        self.model = N_beats(stack_configs, backcast_length, forecast_length)
        self.model.to(self.device)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
        self.is_trained = False
        self.scaler = None
        self.training_stats = {}
        self.loss_functions = {
            'mse': LossCalculator.mse_loss,
            'mae': LossCalculator.mae_loss,
            'mape': LossCalculator.mape_loss,
            'smape': LossCalculator.smape_loss,
            'huber': LossCalculator.huber_loss
        }
        
    def _prepare_data(self, data: np.ndarray, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare time series data for training"""
        if normalize and self.scaler is None:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        elif normalize and self.scaler is not None:
            data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - self.backcast_length - self.forecast_length + 1):
            X.append(data[i:i + self.backcast_length])
            y.append(data[i + self.backcast_length:i + self.backcast_length + self.forecast_length])
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def fit(self, 
            train_data: np.ndarray,
            val_data: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 1e-3,
            optimizer: str = 'adam',
            loss_function: str = 'mse',
            early_stopping: bool = True,
            patience: int = 10,
            normalize: bool = True,
            scheduler: str = 'plateau',
            gradient_clip: float = 1.0,
            verbose: bool = True) -> Dict:
        """
        Train the N-BEATS model
        
        Parameters:
        -----------
        train_data : np.ndarray
            Training time series data
        val_data : np.ndarray, optional
            Validation time series data
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        optimizer : str
            Optimizer choice ('adam', 'adamw', 'sgd', 'rmsprop')
        loss_function : str
            Loss function ('mse', 'mae', 'mape', 'smape', 'huber')
        early_stopping : bool
            Whether to use early stopping
        patience : int
            Early stopping patience
        normalize : bool
            Whether to normalize the data
        scheduler : str
            Learning rate scheduler ('plateau', 'step', 'cosine')
        gradient_clip : float
            Gradient clipping value
        verbose : bool
            Whether to print training progress
        """
        
        # Prepare data
        X_train, y_train = self._prepare_data(train_data, normalize)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data is not None:
            X_val, y_val = self._prepare_data(val_data, normalize)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize optimizer
        optimizers = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop
        }
        
        if optimizer not in optimizers:
            raise ValueError(f"Optimizer {optimizer} not supported. Choose from {list(optimizers.keys())}")
        
        if optimizer == 'sgd':
            opt = optimizers[optimizer](self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            opt = optimizers[optimizer](self.model.parameters(), lr=learning_rate)
        
        # Initialize scheduler
        if scheduler == 'plateau':
            sched = ReduceLROnPlateau(opt, mode='min', patience=patience//2, factor=0.5)
        elif scheduler == 'step':
            sched = StepLR(opt, step_size=epochs//4, gamma=0.5)
        elif scheduler == 'cosine':
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        else:
            sched = None
        
        # Initialize loss function
        if loss_function not in self.loss_functions:
            raise ValueError(f"Loss function {loss_function} not supported. Choose from {list(self.loss_functions.keys())}")
        
        criterion = self.loss_functions[loss_function]
        
        # Initialize early stopping
        early_stopper = EarlyStopping(patience=patience) if early_stopping else None
        
        # Training loop
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': [], 'epoch_times': []}
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                opt.zero_grad()
                predictions = self.model(batch_x)
                loss = criterion(batch_y, predictions)
                loss.backward()
                
                # Gradient clipping
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                opt.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            avg_val_loss = 0.0
            if val_data is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        predictions = self.model(batch_x)
                        loss = criterion(batch_y, predictions)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                self.history['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            if sched is not None:
                if scheduler == 'plateau' and val_data is not None:
                    sched.step(avg_val_loss)
                elif scheduler != 'plateau':
                    sched.step()
            
            # Record learning rate
            current_lr = opt.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Record epoch time
            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)
            
            # Verbose output
            if verbose and (epoch + 1) % 10 == 0:
                if val_data is not None:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f} - LR: {current_lr:.2e} - Time: {epoch_time:.2f}s")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - LR: {current_lr:.2e} - Time: {epoch_time:.2f}s")
            
            # Early stopping
            if early_stopper is not None and val_data is not None:
                if early_stopper(avg_val_loss, self.model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        
        # Store training statistics
        self.training_stats = {
            'total_epochs': len(self.history['train_loss']),
            'best_train_loss': min(self.history['train_loss']),
            'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else None,
            'total_training_time': total_time,
            'avg_epoch_time': np.mean(self.history['epoch_times']),
            'final_lr': self.history['learning_rate'][-1],
            'optimizer': optimizer,
            'loss_function': loss_function,
            'batch_size': batch_size,
            'gradient_clip': gradient_clip
        }
        
        self.is_trained = True
        
        if verbose:
            print(f"\nTraining completed in {total_time:.2f}s")
            print(f"Best train loss: {self.training_stats['best_train_loss']:.6f}")
            if self.training_stats['best_val_loss']:
                print(f"Best val loss: {self.training_stats['best_val_loss']:.6f}")
        
        return self.history
    
    def forecast(self, 
                 input_sequence: np.ndarray,
                 return_components: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Generate forecast for given input sequence
        
        Parameters:
        -----------
        input_sequence : np.ndarray
            Input sequence of length backcast_length
        return_components : bool
            Whether to return individual stack contributions
        
        Returns:
        --------
        forecast : np.ndarray
            Forecasted values
        components : dict (if return_components=True)
            Individual stack contributins
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        if len(input_sequence) != self.backcast_length:
            raise ValueError(f"Input sequence length must be {self.backcast_length}")
        
        # Normalize input if scaler was used
        if self.scaler is not None:
            input_sequence = self.scaler.transform(input_sequence.reshape(-1, 1)).flatten()
        
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            if return_components:
                # Get individual stack contributions
                components = {}
                residual = x
                total_forecast = torch.zeros(1, self.forecast_length, device=self.device)
                
                for i, stack in enumerate(self.model.stacks):
                    stack_forecast, residual = stack(residual)
                    total_forecast += stack_forecast
                    
                    # Denormalize if needed
                    if self.scaler is not None:
                        stack_contribution = self.scaler.inverse_transform(
                            stack_forecast.squeeze().cpu().numpy().reshape(-1, 1)
                        ).flatten()
                    else:
                        stack_contribution = stack_forecast.squeeze().cpu().numpy()
                    
                    stack_config = self.stack_configs[i]
                    stack_name = f"Stack_{i+1}_{stack_config['basis_type']}"
                    components[stack_name] = stack_contribution
                
                forecast = total_forecast.squeeze().cpu().numpy()
            else:
                forecast = self.model(x).squeeze().cpu().numpy()
        
        # Denormalize forecast
        if self.scaler is not None:
            forecast = self.scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
        
        if return_components:
            return forecast, components
        else:
            return forecast
    
    def stats(self) -> Dict:
        """
        Get comprehensive model and training statistics
        
        Returns:
        --------
        stats : dict
            Complete statistics about the model and training
        """
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        # Model architecture info
        model_info = self.model.get_model_info()
        
        # Parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Training statistics
        stats = {
            'model_architecture': model_info,
            'parameters': {
                'total': total_params,
                'trainable': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024) 
            },
            'training_stats': self.training_stats,
            'data_info': {
                'backcast_length': self.backcast_length,
                'forecast_length': self.forecast_length,
                'normalized': self.scaler is not None,
                'device': self.device
            }
        }
        
        # Add training performance metrics
        if self.history['train_loss']:
            stats['performance'] = {
                'train_loss_improvement': (self.history['train_loss'][0] - self.history['train_loss'][-1]) / self.history['train_loss'][0] * 100,
                'convergence_rate': len(self.history['train_loss']) / self.training_stats['total_epochs']
            }
            
            if self.history['val_loss']:
                stats['performance']['val_loss_improvement'] = (self.history['val_loss'][0] - self.history['val_loss'][-1]) / self.history['val_loss'][0] * 100

        print("\n=== Training Statistics ===")
        print(f"Total parameters: {stats['parameters']['total']:,}")
        print(f"Model size: {stats['parameters']['model_size_mb']:.2f} MB")
        print(f"Training time: {stats['training_stats']['total_training_time']:.2f}s")
        print(f"Best train loss: {stats['training_stats']['best_train_loss']:.6f}")
        print(f"Best val loss: {stats['training_stats']['best_val_loss']:.6f}")
        return stats
    
    def plot_training_history(self, figsize: Tuple[int, int] = (15, 5)):
        """Plot training history"""
        if not self.is_trained:
            print("Model not trained yet")
            return      
        fig, axes = plt.subplots(1, 3, figsize=figsize)        
        axes[0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)     
        axes[1].plot(self.history['learning_rate'], color='green')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(self.history['epoch_times'], color='orange')
        axes[2].set_title('Epoch Training Time')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Time (seconds)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_forecast(self, 
                     historical_data: np.ndarray,
                     forecast_data: Optional[np.ndarray] = None,
                     plot_components: bool = True,
                     figsize: Tuple[int, int] = (15, 8)):
        """
        Plot forecast with stack contributions
        
        Parameters:
        -----------
        historical_data : np.ndarray
            Historical time series data (at least backcast_length points)
        forecast_data : np.ndarray, optional
            True future values for comparison
        plot_components : bool
            Whether to plot individual stack contributions
        figsize : tuple
            Figure size
        """
        if not self.is_trained:
            print("Model must be trained before plotting forecast")
            return
        
        if len(historical_data) < self.backcast_length:
            raise ValueError(f"Historical data must have at least {self.backcast_length} points")
        
        # Get forecast and components
        input_seq = historical_data[-self.backcast_length:]
        forecast, components = self.forecast(input_seq, return_components=True)
        
        # Create time indices
        hist_time = np.arange(len(historical_data))
        forecast_time = np.arange(len(historical_data), len(historical_data) + self.forecast_length)
        
        if plot_components:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        
        # Main forecast plot
        ax1.plot(hist_time, historical_data, label='Historical', color='blue', linewidth=2)
        ax1.plot(forecast_time, forecast, label='Forecast', color='red', linewidth=2, linestyle='--')
        
        if forecast_data is not None:
            ax1.plot(forecast_time, forecast_data, label='True Future', color='green', linewidth=2, alpha=0.7)
        input_time = hist_time[-self.backcast_length:]
        ax1.fill_between(input_time, 
                        np.min(historical_data[-self.backcast_length:]), 
                        np.max(historical_data[-self.backcast_length:]), 
                        alpha=0.2, color='yellow', label='Input Sequence')
        
        ax1.set_title('N-BEATS Forecast')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if plot_components:
            colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
            
            for i, (stack_name, contribution) in enumerate(components.items()):
                ax2.plot(forecast_time, contribution, label=stack_name, 
                        color=colors[i], linewidth=2, marker='o', markersize=4)
            
            ax2.plot(forecast_time, forecast, label='Total Forecast', 
                    color='black', linewidth=3, linestyle='--', alpha=0.8)
            
            ax2.set_title('Individual Stack Contributions')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Contribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, test_data: np.ndarray, metrics: List[str] = ['mae', 'mse', 'mape']) -> Dict:
        """
        Evaluate model on test data
        
        Parameters:
        -----------
        test_data : np.ndarray
            Test time series data
        metrics : list
            List of metrics to compute
        
        Returns:
        --------
        results : dict
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        X_test, y_test = self._prepare_data(test_data, normalize=True)
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for i in range(len(X_test)):
                x = X_test[i:i+1].to(self.device)
                pred = self.model(x).cpu().numpy().flatten()
                predictions.extend(pred)
                actuals.extend(y_test[i].numpy())
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            actuals = self.scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
        results = {}
        for metric in metrics:
            if metric == 'mae':
                results['MAE'] = mean_absolute_error(actuals, predictions)
            elif metric == 'mse':
                results['MSE'] = mean_squared_error(actuals, predictions)
            elif metric == 'rmse':
                results['RMSE'] = np.sqrt(mean_squared_error(actuals, predictions))
            elif metric == 'mape':
                results['MAPE'] = np.mean(np.abs((actuals - predictions) / (np.abs(actuals) + 1e-8))) * 100
            elif metric == 'smape':
                results['SMAPE'] = np.mean(2 * np.abs(actuals - predictions) / (np.abs(actuals) + np.abs(predictions) + 1e-8)) * 100
        print("\n=== Evaluation Results ===")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        return results
