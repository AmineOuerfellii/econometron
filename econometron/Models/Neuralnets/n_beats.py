import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import copy
from typing import Dict, List, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchinfo import summary as torch_summary
from econometron.utils.data_preparation.scaler import RevIN
import pandas as pd
import logging
import uuid
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class GENERIC_BASE(nn.Module):
    def __init__(self, back_len, fore_len):
        super(GENERIC_BASE, self).__init__()
        self.back_len = back_len
        self.fore_len = fore_len

    def forward(self, theta_b, theta_f):
        backcast = theta_b
        forecast = theta_f
        return backcast, forecast


class SEASONALITY_FOU(nn.Module):
    def __init__(self, backcast_length, forecast_length, harmonics: int = 1):
        super(SEASONALITY_FOU, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.harmonics = harmonics
        t_backcast = torch.arange(backcast_length).float() / backcast_length
        t_forecast = torch.arange(forecast_length).float() / forecast_length
        self.S_backcast = self._fourier_basis(t_backcast, harmonics)
        self.S_forecast = self._fourier_basis(t_forecast, harmonics)

    def _fourier_basis(self, t: torch.Tensor, H: int):
        max_harmonic = int(np.floor(H / 2))
        basis = [torch.ones_like(t)]
        for i in range(1, max_harmonic):
            basis.append(torch.cos(2 * np.pi * i * t))
        for i in range(1, max_harmonic):
            basis.append(torch.sin(2 * np.pi * i * t))
        return torch.stack(basis, dim=1)

    def forward(self, theta_b, theta_f):
        backcast = torch.matmul(theta_b, self.S_backcast.T)
        forecast = torch.matmul(theta_f, self.S_forecast.T)
        return backcast, forecast


class TREND_BASIS(nn.Module):
    def __init__(self, backcast_length: int, forecast_length: int, degree: int = 2, basis_type: str = "poly"):
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.degree = degree
        self.basis_type = basis_type
        if basis_type == "poly":
            t_back = torch.arange(-backcast_length,
                                  0).float() / forecast_length
            t_fore = torch.arange(0, forecast_length).float() / forecast_length
            self.register_buffer(
                "backcast_basis", self._poly_basis(t_back, degree))
            self.register_buffer(
                "forecast_basis", self._poly_basis(t_fore, degree))
        elif basis_type == "cheb":
            t_back = np.linspace(-1, 1, backcast_length)
            t_fore = np.linspace(-1, 1, forecast_length)
            basis_back = self._chebyshev_basis(t_back, degree)
            basis_fore = self._chebyshev_basis(t_fore, degree)
            self.register_buffer("backcast_basis", torch.tensor(
                basis_back, dtype=torch.float32))
            self.register_buffer("forecast_basis", torch.tensor(
                basis_fore, dtype=torch.float32))
        else:
            raise ValueError("basis_type must be 'poly' or 'cheb'")

    def _poly_basis(self, t: torch.Tensor, degree: int):
        return torch.stack([t**i for i in range(degree+1)], dim=1).T

    def _chebyshev_basis(self, t: np.ndarray, degree: int):
        N = len(t)
        basis = np.zeros((degree+1, N))
        basis[0, :] = 1.0
        if degree >= 1:
            basis[1, :] = t
        for n in range(2, degree+1):
            basis[n, :] = 2 * t * basis[n-1, :] - basis[n-2, :]
        return basis

    def forward(self, theta_backcast, theta_forecast):
        forecast = torch.matmul(theta_forecast, self.forecast_basis)
        backcast = torch.matmul(theta_backcast, self.backcast_basis)
        return backcast, forecast


class NBEATS_BLOCK(nn.Module):
    def __init__(self, n: int = 1, H: int = 1, basis: str = 'GENERIC', b_type='poly', Harmonics: int = 2, Degree: int = 2, Dropout: int = 0.1, Layer_size: int = 512, num_lay_per_B: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        super(NBEATS_BLOCK, self).__init__()
        if n <= 1 or H <= 1:
            raise ValueError('n and H must be greater than 1')
        self.Backcast_len = n*H
        self.forecast_len = H
        self.basis = basis
        self.Harmonics = Harmonics
        self.Degree = Degree
        self.Dropout = Dropout
        self.b_type = b_type
        self.Layer_size = Layer_size
        layers = [nn.Linear(in_features=self.Backcast_len,
                            out_features=self.Layer_size), nn.ReLU(), nn.Dropout(Dropout)]
        for _ in range(num_lay_per_B - 2):
            layers.append(nn.Linear(in_features=self.Layer_size,
                          out_features=self.Layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Dropout))
        layers.append(nn.Linear(in_features=self.Layer_size,
                      out_features=self.Layer_size))
        layers.append(nn.ReLU())
        self.FC_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        def _generic_block(x: torch.Tensor):
            T_b = nn.Linear(in_features=self.Layer_size,
                            out_features=self.Backcast_len)
            T_f = nn.Linear(in_features=self.Layer_size,
                            out_features=self.forecast_len)
            back, fore = GENERIC_BASE(
                self.Backcast_len, self.forecast_len)(T_b(x), T_f(x))
            return fore, back

        def _Trend_Block(x: torch.Tensor, degree: int, basis: str):
            T_b = nn.Linear(in_features=self.Layer_size,
                            out_features=self.Degree+1)
            T_f = nn.Linear(in_features=self.Layer_size,
                            out_features=self.Degree+1)
            back, fore = TREND_BASIS(
                self.Backcast_len, self.forecast_len, self.Degree, self.b_type)(T_b(x), T_f(x))
            return fore, back

        def _Seasonality_Block(x: torch.Tensor):
            num_basis = 2 * int(np.floor(self.Harmonics / 2)) - 1
            T_b = nn.Linear(in_features=self.Layer_size,
                            out_features=num_basis)
            T_f = nn.Linear(in_features=self.Layer_size,
                            out_features=num_basis)
            back, fore = SEASONALITY_FOU(
                self.Backcast_len, self.forecast_len, self.Harmonics)(T_b(x), T_f(x))
            return fore, back
        x_ = self.FC_stack(x)
        fore, back = None, None
        if self.basis == 'GENERIC':
            fore, back = _generic_block(x_)
        elif self.basis == 'TREND':
            fore, back = _Trend_Block(x_, self.Degree, self.basis)
        elif self.basis == 'SEASONALITY':
            fore, back = _Seasonality_Block(x_)
        return fore, back


class NBEATS_STACK(nn.Module):
    def __init__(self, num_B_per_S: int = 3, n_h: int = 1, H: int = 1, Blocks: list = ['G'], Harmonics: list = [2], Degree: list = [2], Dropout: list = [0.1], Layer_size: list = [512], num_lay_per_B: list = [3], share_weights: bool = False):
        super(NBEATS_STACK, self).__init__()
        self.backcast_len = n_h * H
        self.forecast_len = H
        self.num_B_per_S = num_B_per_S
        self.share_weights = share_weights
        default_harmonics = 2
        default_degree = 2
        default_dropout = 0.1
        default_layer_size = 512
        default_num_lay_per_B = 3
        default_block_type = 'G'
        if not Blocks:
            Blocks = [default_block_type] * num_B_per_S
        elif len(Blocks) == 1:
            Blocks = Blocks * num_B_per_S
        elif len(Blocks) < num_B_per_S:
            logger.warning(
                f"Blocks list length {len(Blocks)} is less than num_B_per_S {num_B_per_S}. Filling with 'G'.")
            Blocks = Blocks + [default_block_type] * \
                (num_B_per_S - len(Blocks))
        elif len(Blocks) > num_B_per_S:
            logger.warning(
                f"Blocks list length {len(Blocks)} exceeds num_B_per_S {num_B_per_S}. Taking first {num_B_per_S} elements.")
            Blocks = Blocks[:num_B_per_S]
        if not all(t in ['G', 'T', 'T_P', 'S'] for t in Blocks):
            raise ValueError(
                "Blocks must contain only 'G', 'T', 'T_P', or 'S'")
        self.Blocks = Blocks

        def normalize_list(param_list, default, target_len, name):
            if not param_list:
                return [default] * target_len
            elif len(param_list) == 1:
                return param_list * target_len
            elif len(param_list) < target_len:
                logger.warning(
                    f"{name} list length {len(param_list)} is less than {target_len}. Filling with default {default}.")
                return param_list + [default] * (target_len - len(param_list))
            elif len(param_list) > target_len:
                logger.warning(
                    f"{name} list length {len(param_list)} exceeds {target_len}. Taking first {target_len} elements.")
                return param_list[:target_len]
            return param_list
        self.Layer_size = normalize_list(
            Layer_size, default_layer_size, num_B_per_S, "Layer_size")
        self.Dropout = normalize_list(
            Dropout, default_dropout, num_B_per_S, "Dropout")
        self.num_lay_per_B = normalize_list(
            num_lay_per_B, default_num_lay_per_B, num_B_per_S, "num_lay_per_B")
        num_S_blocks = sum(1 for t in Blocks if t == 'S')
        num_T_blocks = sum(1 for t in Blocks if t in ['T', 'T_P'])
        self.Harmonics = normalize_list(
            Harmonics, default_harmonics, num_S_blocks, "Harmonics")
        self.Degree = normalize_list(
            Degree, default_degree, num_T_blocks, "Degree")
        if n_h < 1 or H < 1:
            raise ValueError('n_h and H must be positive integers')
        if self.backcast_len % H != 0:
            raise ValueError('n_h * H must be consistent with H')
        self.blocks = nn.ModuleList()
        s_index = 0
        t_index = 0
        if self.share_weights:
            block_dict = {}
            for i, block_type in enumerate(self.Blocks):
                layer_size = self.Layer_size[i]
                dropout = self.Dropout[i]
                num_lay_per_B = self.num_lay_per_B[i]
                harmonics = self.Harmonics[s_index] if block_type == 'S' else default_harmonics
                degree = self.Degree[t_index] if block_type in [
                    'T', 'T_C'] else default_degree
                if block_type == 'S':
                    basis = 'SEASONALITY'
                    basis_type = None
                    s_index += 1
                elif block_type in ['T', 'T_C']:
                    basis = 'TREND'
                    basis_type = 'poly' if block_type == 'T_C' else 'cheb'
                    t_index += 1
                else:
                    basis = 'GENERIC'
                    basis_type = None
                block_key = (block_type, harmonics, degree,
                             dropout, layer_size, num_lay_per_B)
                if block_key not in block_dict:
                    block_dict[block_key] = NBEATS_BLOCK(n=n_h, H=self.forecast_len, basis=basis, Harmonics=harmonics, Degree=degree, Dropout=dropout, Layer_size=layer_size, num_lay_per_B=num_lay_per_B,
                                                         b_type=basis_type)
                self.blocks.append(block_dict[block_key])
        else:
            for i, block_type in enumerate(self.Blocks):
                layer_size = self.Layer_size[i]
                dropout = self.Dropout[i]
                num_lay_per_B = self.num_lay_per_B[i]
                harmonics = self.Harmonics[s_index] if block_type == 'S' else default_harmonics
                degree = self.Degree[t_index] if block_type in [
                    'T', 'T_C'] else default_degree
                if block_type == 'S':
                    basis = 'SEASONALITY'
                    basis_type = None
                    s_index += 1
                elif block_type in ['T', 'T_C']:
                    basis = 'TREND'
                    basis_type = 'poly' if block_type == 'T_C' else 'cheb'
                    t_index += 1
                else:
                    basis = 'GENERIC'
                    basis_type = None
                block = NBEATS_BLOCK(n=n_h, H=self.forecast_len, basis=basis, Harmonics=harmonics, Degree=degree,
                                     Dropout=dropout, Layer_size=layer_size, num_lay_per_B=num_lay_per_B, b_type=basis_type)
                self.blocks.append(block)

    def forward(self, x: torch.Tensor):
        if self.num_B_per_S == 0:
            raise ValueError("Number of blocks must be greater than 0")
        stack_forecast = torch.zeros(
            x.shape[0], self.forecast_len, device=x.device)
        residual = x
        for block in self.blocks:
            fore, back = block(residual)
            residual = residual-back
            stack_forecast += fore
        return stack_forecast, residual


class NBEATS(nn.Module):
    def __init__(self, n: int, h: int, n_s: int, stack_configs: list):
        super(NBEATS, self).__init__()
        self.n_features = 1
        self.revin = RevIN(num_features=self.n_features)
        self.backcast_length = n * h
        self.forecast_length = h
        self.num_stacks = n_s
        default_config = {'num_B_per_S': 3, 'Blocks': ['G'], 'Harmonics': [2], 'Degree': [
            2], 'Dropout': [0.1], 'Layer_size': [512], 'num_lay_per_B': [3], 'share_weights': False}
        if not stack_configs:
            logger.warning(
                f"No stack_configs provided. Using default configuration for {n_s} stacks.")
            stack_configs = [default_config] * n_s
        elif len(stack_configs) < n_s:
            logger.warning(
                f"stack_configs length {len(stack_configs)} is less than n_s {n_s}. Filling with default configuration.")
            stack_configs = stack_configs + \
                [default_config] * (n_s - len(stack_configs))
        elif len(stack_configs) > n_s:
            logger.warning(
                f"stack_configs length {len(stack_configs)} exceeds n_s {n_s}. Taking first {n_s} elements.")
            stack_configs = stack_configs[:n_s]
        for i, config in enumerate(stack_configs):
            if not isinstance(config, dict):
                raise ValueError(
                    f"stack_configs[{i}] must be a dictionary, got {type(config)}")
            if 'Blocks' not in config:
                logger.warning(
                    f"stack_configs[{i}] missing 'Blocks'. Using default ['G'].")
                config['Blocks'] = default_config['Blocks']
            if not all(t in ['G', 'T', 'T_P', 'S'] for t in config['Blocks']):
                raise ValueError(
                    f"stack_configs[{i}] contains invalid block types. Must be 'G', 'T', 'T_P', or 'S'.")
        self.stacks = nn.ModuleList()
        for config in stack_configs:
            self.stacks.append(NBEATS_STACK(num_B_per_S=config.get('num_B_per_S', default_config['num_B_per_S']), n_h=n, H=self.forecast_length, Blocks=config.get('Blocks', default_config['Blocks']),
                                            Harmonics=config.get('Harmonics', default_config['Harmonics']), Degree=config.get('Degree', default_config['Degree']), Dropout=config.get('Dropout', default_config['Dropout']),
                                            Layer_size=config.get('Layer_size', default_config['Layer_size']), num_lay_per_B=config.get('num_lay_per_B', default_config['num_lay_per_B']),
                                            share_weights=config.get('share_weights', default_config['share_weights'])))

    def forward(self, x: torch.Tensor, mode: str = None, cas: str = None):
        stats = None
        if mode == 'norm':
            x, stats = self.revin(x, "norm")
            x = x.squeeze(-1)
        # print(x.shape)
        total_forecast = torch.zeros(
            x.shape[0], self.forecast_length, device=x.device)
        residual = x
        for stack in self.stacks:
            fore, residual = stack(residual)
            total_forecast += fore
        if mode == 'norm' and cas == 'fore':
            tot = total_forecast, stats
            # print(total_forecast)
            # print(stats)
            # print(total_forecast.shape)
            # print(self.revin(tot, "denorm").shape)
            total_forecast = self.revin(tot, "denorm")
            # print(total_forecast.shape)
        return total_forecast
#######################################
# MAPE LOSS


class MAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(MAPELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.mean(
            torch.abs((y_true - y_pred) / (y_true + self.eps))) * 100
        return loss
#######################################
# Trainer class
######################################


class Trainer_ts:
    def __init__(self, model, normalization_type: str = None, device: str = None, Seed: int = 42):
        if not isinstance(model, nn.Module):
            raise ValueError("Model must be a PyTorch nn.Module instance")
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        if normalization_type not in ["revin", "local", "global", None]:
            logger.info(
                f"Normalization needs to be either RevIN, Local, Global, or None. Falling back to None")
            normalization_type = None
        self.normalization_type = normalization_type
        self.device = device
        self.model.to(device)
        self.model_initial_state = copy.deepcopy(self.model.state_dict())
        self.seed = Seed
        #############
        torch.manual_seed(self.seed)
        if device == 'cuda':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)
        ##############
        self.data = None
        self.Back_coeff = None
        self.forecast = None
        self.training_hist = {'train_loss': [],
                              'val_loss': [], 'lr': [], 'epochs': []}
        self.val_split = None
        self.stats_train = None
        self.stats_val = None
        self._setup_logging()

    def _setup_logging(self):
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def _validate_data(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if isinstance(data, pd.DataFrame):
                if data.shape[1] > 1:
                    logger.warning(
                        f"DataFrame has {data.shape[1]} columns. Using first column.")
                    data = data.iloc[:, 0]
                data = data.values
        elif not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                raise ValueError(f"Could not convert data to numpy array: {e}")
        data = np.atleast_1d(data).flatten()
        if np.isnan(data).any():
            nan_count = np.isnan(data).sum()
            logger.warning(f"Found {nan_count} NaN values. Removing them.")
            data = data[~np.isnan(data)]
        if np.isinf(data).any():
            inf_count = np.isinf(data).sum()
            logger.warning(
                f"Found {inf_count} infinite values. Removing them.")
            data = data[~np.isinf(data)]
        if len(data) == 0:
            raise ValueError("No valid data points after cleaning")
        data_range = np.max(data) - np.min(data)
        if data_range > 1e10:
            logger.warning(
                f"Data range is very large ({data_range:.2e}). Consider scaling your data.")
        return data.astype(np.float32)

    def _create_windows(self, data: np.ndarray, n: int, forecast_len: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(data) < (1+n)*forecast_len:
            raise ValueError(
                f"Data length ({len(data)}) must be at least "
                f"backcast_length + forecast_length ({(1+n)*forecast_len})")
        n_r = len(data)-forecast_len*(n+1)+1
        X = np.zeros((n_r, forecast_len*n))
        Y = np.zeros((n_r, forecast_len))
        for i in range(len(data)-forecast_len*(n+1)+1):
            X[i, :] = data[i:i+n*forecast_len]
            Y[i, :] = data[i+n*forecast_len:i+(1+n)*forecast_len]
        logger.info(f"Creted sequences - X shape {X.shape} ,y shape {Y.shape}")
        return X, Y

    def _local_normalize(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        stats = np.zeros((X.shape[0], 4), dtype=np.float32)
        X_norm = np.zeros_like(X, dtype=np.float32)
        y_norm = np.zeros_like(y, dtype=np.float32)
        for i in range(X.shape[0]):
            mean_X, std_X = X[i].mean(), X[i].std() + 1e-8
            mean_y, std_y = y[i].mean(), y[i].std() + 1e-8
            stats[i] = [mean_X, std_X, mean_y, std_y]
            X_norm[i] = (X[i] - mean_X) / std_X
            y_norm[i] = (y[i] - mean_y) / std_y
        return X_norm, y_norm, stats

    def MapeLoss(self, output, target, epsilon=1e-8):
        absolute_error = torch.abs(target - output)
        percentage_error = absolute_error / (torch.abs(target) + epsilon)
        return torch.mean(percentage_error) * 100

    def _local_denormalize(self, X: torch.Tensor, stats: np.ndarray, Y: torch.Tensor = None) -> Tuple[np.ndarray, np.ndarray]:
        X_norm = X.detach().cpu().numpy().copy()
        Y_norm = Y.detach().cpu().numpy().copy() if Y is not None else None
        mean_X, std_X, mean_y, std_y = stats[:,
                                             0], stats[:, 1], stats[:, 2], stats[:, 3]
        X_den = X_norm * std_X.reshape(-1, 1) + mean_X.reshape(-1, 1)
        Y_den = Y_norm * \
            std_y.reshape(-1, 1) + mean_y.reshape(-1,
                                                  1) if Y_norm is not None else None
        return X_den, Y_den

    def _global_normalize(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        else:
            X = np.asarray(X)
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        else:
            y = np.asarray(y)
        series = np.asarray(self.data).ravel()
        mu = np.mean(series)
        sigma = np.std(series) + 1e-8
        stats = {"mean": mu, "std": sigma}
        X_norm = (X - mu) / sigma
        y_norm = (y - mu) / sigma
        return X_norm.astype(np.float32), y_norm.astype(np.float32), stats

    def _global_denormalize(self, predictions: torch.Tensor, stats: dict) -> np.ndarray:
        preds = predictions.detach().cpu().numpy().copy()
        denorm = preds * stats["std"] + stats["mean"]
        return denorm

    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame, pd.Series], n: int, forecast_len: int, val_split: float = 0.2) -> Tuple[TensorDataset, TensorDataset]:
        data = self._validate_data(data)
        self.data = data.copy()
        train_stats = None
        val_stats = None
        logger.info(f"Preparing datasets with {len(data)} data points")
        X, y = self._create_windows(data, n, forecast_len)
        if len(X) < 10:
            logger.warning(
                f"Only {len(X)} sequences created. Consider using shorter backcast/forecast lengths.")
        split_idx = max(1, int(len(X)*(1-val_split)))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        logger.info(
            f"Train sequences: {len(X_train)}, Validation sequences: {len(X_val)}")
        if self.normalization_type == 'local':
            X_train, y_train, train_stats = self._local_normalize(
                X_train, y_train)
            X_val, y_val, val_stats = self._local_normalize(X_val, y_val)
        if self.normalization_type == 'global':
            X_train, y_train, train_stats = self._global_normalize(
                X_train, y_train)
            X_val, y_val, val_stats = self._global_normalize(X_val, y_val)
        train_dataset = TensorDataset(torch.from_numpy(
            X_train).float(), torch.from_numpy(y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(
            X_val).float(), torch.from_numpy(y_val).float())
        return train_dataset, val_dataset, train_stats, val_stats

    def fit(self, Data: Union[np.ndarray, pd.DataFrame, pd.Series], N: int, Horizon: int, max_epochs: int = 100, optimizer: str = 'adam', lr: float = 1e-4, batch_size: int = 32, grad_clip: float = 1, scheduler: str = 'plateau', loss_fun: str = 'mae',
            early_stopping: int = 20, val_split: float = 0.2, verbose: bool = True) -> Dict:
        self.forecast = Horizon
        self.back_coef = N
        self.val_s = val_split
        self.model.load_state_dict(self.model_initial_state)
        from tqdm import tqdm
        logger.info(f"{'='*30} Model Training - START - {'='*30}")
        self.training_history = {'train_loss': [],
                                 'val_loss': [], 'lr': [], 'epoch': []}
        train_dataset, val_dataset, stats_train, val_stats = self._prepare_data(
            Data, N, Horizon, val_split)
        if len(train_dataset) == 0:
            raise ValueError("No training sequences available")
        if len(val_dataset) == 0:
            logger.warning(
                "No validation sequences available. Using training dataset for validation.")
            val_dataset = train_dataset
        self.stats_train = stats_train
        self.stats_val = val_stats
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        optimizers = {"adam": torch.optim.Adam,
                      "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
        criterions = {'mse': nn.MSELoss(), 'mae': nn.L1Loss(), 'huber': nn.HuberLoss(
        ), 'smooth_l1': nn.SmoothL1Loss(), 'mape': MAPELoss()}
        schedulers = {'plateau': ReduceLROnPlateau,
                      'cosine': CosineAnnealingLR, 'step': torch.optim.lr_scheduler.StepLR}
        opt_class = optimizers.get(optimizer.lower(), torch.optim.Adam)
        optimizer = opt_class(self.model.parameters(), lr=lr)
        criterion = criterions.get(loss_fun.lower(), nn.L1Loss())
        scheduler_class = schedulers.get(scheduler.lower())
        if scheduler_class == ReduceLROnPlateau:
            # Make these params configurable if needed
            scheduler = scheduler_class(
                optimizer, mode='min', factor=0.5, patience=3)
        elif scheduler_class == CosineAnnealingLR:
            scheduler = scheduler_class(
                optimizer, T_max=max_epochs, eta_min=1e-7)
        elif scheduler_class == torch.optim.lr_scheduler.StepLR:
            scheduler = scheduler_class(
                optimizer, step_size=max_epochs // 3, gamma=0.1)
        else:
            logger.warning(
                f"Unknown scheduler '{scheduler}'. No scheduler will be used.")
            scheduler = None
        best_val_loss = float('inf')
        epochs_without_improve = 0
        self.best_model_state = None
        for epoch in tqdm(range(max_epochs), desc="Training Epochs", disable=not verbose):
            self.model.train()
            train_loss = 0
            train_batches = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                if self.normalization_type == 'revin':
                    self.model.n_features = N*self.forecast
                    outputs = self.model(inputs, mode='norm', cas='fore')
                else:
                    outputs = self.model(inputs)
                loss = criterion(outputs, targets.squeeze(-1))
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(
                        f"Invalid loss at epoch {epoch+1}, train batch {batch_idx+1}")
                    continue
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_batches += 1
            if train_batches == 0:
                logger.error("No valid training batches. Stopping training.")
                break
            epoch_train_loss = train_loss / len(train_loader.dataset)
            # Validation
            self.model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    if self.normalization_type == 'revin':
                        # targets, tar_stat = self.model.revin(targets, mode='norm')
                        self.model.n_features = N*self.forecast
                        outputs = self.model(inputs, mode='norm', cas='fore')
                    else:
                        outputs = self.model(inputs)
                    loss = criterion(outputs, targets.squeeze(-1))
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(
                            f"Invalid loss at epoch {epoch+1}, val batch {batch_idx+1}")
                        continue
                    val_loss += loss.item() * inputs.size(0)
                    val_batches += 1
            if val_batches == 0:
                logger.warning(
                    f"No valid validation batches at epoch {epoch+1}. Skipping validation metrics.")
                epoch_val_loss = float('inf')
            else:
                epoch_val_loss = val_loss / len(val_loader.dataset)
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_without_improve = 0
                self.best_model_state = self.model.state_dict()
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= early_stopping:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            current_lr = optimizer.param_groups[0]['lr']
            self.training_history['lr'].append(current_lr)
            self.training_history['train_loss'].append(epoch_train_loss)
            self.training_history['val_loss'].append(epoch_val_loss)
            self.training_history['epoch'].append(epoch + 1)
            if verbose:
                logger.info(
                    f"Epoch {epoch+1}/{max_epochs} - train_loss: {epoch_train_loss:.4f} - val_loss: {epoch_val_loss:.4f} - lr: {current_lr:.6f}")
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(
                f"Restored best model state with validation loss: {best_val_loss:.6f}")
        else:
            logger.warning("No best model state saved")
        logger.info(f"{'='*30} Model Training - END - {'='*30}")
        return self.training_history

    def find_optimal_lr(self, data, back_coeff=1, Horizon=1, val_split=0.2, batch_size: int = 32, start_lr=1e-7, end_lr=10, num_iter=100, restore_weights=True, optimizer='adam', loss_fun='mae', plot=True):
        logger.info(
            f"Starting learning rate finder from {start_lr} to {end_lr}")
        logger.info(f"Using optimizer: {optimizer}, loss function: {loss_fun}")
        if restore_weights:
            original_state = self.model_initial_state
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        train_dataset, val_dataset, stats_train, stats_val = self._prepare_data(
            data, back_coeff, Horizon, val_split)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        optimizers = {"adam": torch.optim.Adam,
                      "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
        criterions = {'mse': nn.MSELoss(), 'mae': nn.L1Loss(), 'huber': nn.HuberLoss(
        ), 'smooth_l1': nn.SmoothL1Loss(), 'mape': self.MapeLoss}
        optimizer_class = optimizers.get(optimizer.lower(), optimizers['adam'])
        optimizer = optimizer_class(self.model.parameters(), lr=start_lr)
        criterion = criterions.get(loss_fun.lower(), criterions['mae'])
        mult_factor = (end_lr / start_lr) ** (1 / num_iter)
        learning_rates = []
        losses = []
        best_loss = float('inf')
        self.model.train()
        train_iter = iter(train_loader)
        for iteration in range(num_iter):
            try:
                try:
                    inputs, targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    inputs, targets = next(train_iter)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                current_lr = optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)
                optimizer.zero_grad()
                if self.normalization_type == 'revin':
                    self.model.n_features = back_coeff*Horizon
                    outputs = self.model(inputs, mode='norm', cas='fore')
                    # print('bigo',outputs.shape==(batch_size,Horizon))
                    # targets, tar_stat = self.model.revin(targets, mode='norm')
                else:
                    outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"Invalid loss at iteration {iteration+1}, skipping")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= mult_factor
                    continue
                loss.backward()
                optimizer.step()
                current_loss = loss.item()
                losses.append(current_loss)
                if current_loss < best_loss:
                    best_loss = current_loss
                elif current_loss > best_loss * 4:
                    logger.info(
                        f"Loss exploded at iteration {iteration+1}, stopping early")
                    break
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= mult_factor
                if (iteration + 1) % 20 == 0:
                    logger.info(
                        f"Iteration {iteration+1}/{num_iter}, LR: {current_lr:.2e}, Loss: {current_loss:.6f}")
            except Exception as e:
                logger.error(f"Error at iteration {iteration+1}: {e}")
                break
        if restore_weights and 'original_state' in locals():
            self.model.load_state_dict(original_state)
            logger.info("Model weights restored to original state")
        if len(losses) > 10:
            smoothed_losses = []
            window = min(5, len(losses) // 10)
            for i in range(len(losses)):
                start_idx = max(0, i - window)
                end_idx = min(len(losses), i + window + 1)
                smoothed_losses.append(np.mean(losses[start_idx:end_idx]))
            gradients = np.gradient(smoothed_losses)
            min_gradient_idx = np.argmin(gradients)
            suggested_lr = learning_rates[min_gradient_idx]
            logger.info(f"Suggested learning rate: {suggested_lr:.2e}")
            logger.info(
                f"Learning rate finder completed. Found {len(losses)} valid points.")
            if plot:
                try:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    ax1.plot(learning_rates, losses, 'b-',
                             linewidth=2, label='Loss')
                    ax1.plot(learning_rates, smoothed_losses, 'r--',
                             linewidth=1, alpha=0.7, label='Smoothed Loss')
                    ax1.scatter([suggested_lr], [smoothed_losses[min_gradient_idx]],
                                color='red', s=100, zorder=5, label=f'Optimal LR: {suggested_lr:.2e}')
                    ax1.set_xscale('log')
                    ax1.set_xlabel('Learning Rate')
                    ax1.set_ylabel('Loss')
                    ax1.set_title('Learning Rate Finder - Linear Scale')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                    ax2.plot(learning_rates, losses, 'b-',
                             linewidth=2, label='Loss')
                    ax2.plot(learning_rates, smoothed_losses, 'r--',
                             linewidth=1, alpha=0.7, label='Smoothed Loss')
                    ax2.scatter([suggested_lr], [smoothed_losses[min_gradient_idx]],
                                color='red', s=100, zorder=5, label=f'Optimal LR: {suggested_lr:.2e}')
                    ax2.set_xscale('log')
                    ax2.set_yscale('log')
                    ax2.set_xlabel('Learning Rate')
                    ax2.set_ylabel('Loss (log scale)')
                    ax2.set_title('Learning Rate Finder - Log-Log Scale')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    plt.tight_layout()
                    plt.show()
                    logger.info("Learning rate finder plot displayed")
                except Exception as e:
                    logger.error(f"Error creating plot: {e}")
        else:
            suggested_lr = start_lr
            logger.warning(
                "Not enough valid points for learning rate suggestion")
        return learning_rates, losses, suggested_lr

    def build_rolling_from_series(self, preds, test_series, L_back):
        fc = preds.detach().cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)
        N, H = fc.shape
        idx = test_series.index if hasattr(
            test_series, 'index') else np.arange(len(test_series))
        if hasattr(test_series, 'index') and hasattr(idx, '__getitem__'):
            end_idx = min(len(idx), L_back + (N + H - 1))
            verif_index = idx[L_back: end_idx]
        else:
            end_idx = min(len(test_series), L_back + (N + H - 1))
            verif_index = np.arange(L_back, end_idx)
        T = len(verif_index)
        forecast_latest = np.full(T, np.nan)
        latest_origin = -np.ones(T, int)
        chosen_h = np.full(T, np.nan)
        if hasattr(test_series, 'loc'):
            actual = test_series.loc[verif_index].to_numpy(dtype=float)
        else:
            actual = test_series[verif_index]
        for i in range(N):
            for h in range(H):
                t = i + h
                if t >= T:
                    break
                if i >= latest_origin[t]:
                    forecast_latest[t] = fc[i, h]
                    latest_origin[t] = i
                    chosen_h[t] = h + 1
        return forecast_latest, actual, verif_index

    def summary(self, input_shape=None, plot_training=True, detailed=True, val_split=0.2):
        logger.info("="*50)
        logger.info("MODEL SUMMARY")
        logger.info("="*50)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Normalization: {self.normalization_type}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        if detailed:
            try:
                if input_shape is None:
                    try:
                        input_shape = (1, self.model.backcast_length)
                    except AttributeError:
                        logger.warning(
                            "Could not infer backcast_length. Please provide input_shape.")
                        input_shape = None
                if input_shape:
                    logger.info("\nDetailed Model Architecture:")
                    torch_summary(
                        self.model, input_size=input_shape, device=self.device)
            except Exception as e:
                logger.warning(f"Could not generate detailed summary: {e}")
        if hasattr(self, 'training_history') and self.training_history['train_loss']:
            history = self.training_history
            logger.info("\nTRAINING HISTORY SUMMARY")
            logger.info("-" * 30)
            logger.info(f"Total epochs trained: {len(history['epoch'])}")
            logger.info(
                f"Best training loss: {min(history['train_loss']):.6f}")
            logger.info(
                f"Best validation loss: {min(history['val_loss']):.6f}")
            logger.info(f"Final learning rate: {history['lr'][-1]:.2e}")
        else:
            logger.info("No training history available")
        try:
            backcast_len = self.model.backcast_length
            horizon = self.model.forecast_length
            train_dataset, val_dataset, train_stats, val_stats = self._prepare_data(
                self.data, n=backcast_len // horizon, forecast_len=horizon, val_split=val_split)
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return
        self.model.eval()
        with torch.no_grad():
            X_train, y_train = train_dataset.tensors
            X_val, y_val = val_dataset.tensors
            ######
            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            if self.normalization_type == 'revin':
                train_pred = self.model(X_train, mode='norm', cas='fore')
                val_pred = self.model(X_val, mode='norm', cas='fore')
                train_pred = train_pred.cpu().numpy()
                val_pred = val_pred.cpu().numpy()
                y_train = y_train.cpu().numpy()
                y_val = y_val.cpu().numpy()
            elif self.normalization_type == 'local':
                train_pred = self.model(X_train)
                val_pred = self.model(X_val)
                _, train_pred = self._local_denormalize(
                    X_train, train_stats, train_pred)
                _, val_pred = self._local_denormalize(
                    X_val, val_stats, val_pred)
                _, y_train = self._local_denormalize(
                    X_train, train_stats, y_train)
                _, y_val = self._local_denormalize(X_val, val_stats, y_val)
            elif self.normalization_type == 'global':
                train_pred = self.model(X_train)
                val_pred = self.model(X_val)
                train_pred = self._global_denormalize(train_pred, train_stats)
                val_pred = self._global_denormalize(val_pred, val_stats)
                y_train = self._global_denormalize(y_train, train_stats)
                y_val = self._global_denormalize(y_val, val_stats)
            else:
                train_pred = self.model(X_train).cpu().numpy()
                val_pred = self.model(X_val).cpu().numpy()
                y_train = y_train.cpu().numpy()
                y_val = y_val.cpu().numpy()

        def calculate_metrics(y_true, y_pred):
            y_true_flat = y_true
            y_pred_flat = y_pred

            mae = mean_absolute_error(y_true_flat, y_pred_flat)
            mse = mean_squared_error(y_true_flat, y_pred_flat)
            rmse = np.sqrt(mse)
            mape = np.mean(
                np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
            directional_accuracy = np.mean(np.sign(np.diff(y_true_flat)) == np.sign(
                np.diff(y_pred_flat))) * 100 if len(y_true_flat) > 1 else 0.0
            return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'Directional_Accuracy': directional_accuracy}
        train_metrics = calculate_metrics(y_train, train_pred)
        val_metrics = calculate_metrics(y_val, val_pred)
        logger.info("\nTRAIN PREDICTION STATS")
        logger.info("-" * 30)
        for metric, value in train_metrics.items():
            logger.info(f"{metric}: {value:.2f}%") if 'Accuracy' in metric else logger.info(
                f"{metric}: {value:.4f}")

        logger.info("\nVALIDATION PREDICTION STATS")
        logger.info("-" * 30)
        for metric, value in val_metrics.items():
            logger.info(f"{metric}: {value:.2f}%") if 'Accuracy' in metric else logger.info(
                f"{metric}: {value:.4f}")
        if plot_training:
            try:
                sns.set_style("whitegrid")
                plt.style.use('seaborn-v0_8-bright')
                fig = plt.figure(figsize=(24, 15))
                gs = fig.add_gridspec(3, 3)
                model_name = self.model.__class__.__name__
                if hasattr(self, 'training_history') and self.training_history['train_loss']:
                    history = self.training_history
                    epochs = history['epoch']
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.plot(epochs, history['train_loss'], marker='o',
                             color='#1f77b4', label='Training Loss', linewidth=2)
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.set_title(f'{model_name} Training Loss Curve')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax2 = fig.add_subplot(gs[0, 1])
                    ax2.plot(epochs, history['val_loss'], marker='o',
                             color='#ff7f0e', label='Validation Loss', linewidth=2)
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.set_title(f'{model_name} Validation Loss Curve')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax_lr = fig.add_subplot(gs[0, 2])
                    ax_lr.plot(
                        epochs, history['lr'], marker='o', color='#2ca02c', label='Learning Rate', linewidth=2)
                    ax_lr.set_yscale('log')
                    ax_lr.set_xlabel('Epoch')
                    ax_lr.set_ylabel('Learning Rate')
                    ax_lr.set_title(f'{model_name} Learning Rate Schedule')
                    ax_lr.legend()
                    ax_lr.grid(True, alpha=0.3)
                ax3 = fig.add_subplot(gs[1, :])
                data_series = pd.Series(self.data) if not isinstance(
                    self.data, pd.Series) else self.data
                full_index = data_series.index
                full_values = data_series.values
                actual_index = full_index[backcast_len:]
                actual_values = full_values[backcast_len:]
                train_size = train_pred.shape[0]
                train_forecast, train_actual, train_idx = self.build_rolling_from_series(
                    train_pred, data_series[:train_size], backcast_len)
                val_forecast, val_actual, val_idx = self.build_rolling_from_series(
                    val_pred, data_series[train_size:], backcast_len)
                ax3.plot(actual_index, actual_values, label='Actual',
                         color='#2ca02c', linewidth=2)
                if len(train_forecast) > 0:
                    valid_train_mask = ~np.isnan(train_forecast)
                    ax3.plot(train_idx[valid_train_mask], train_forecast[valid_train_mask],
                             label='Train Predictions', color='#d62728', linewidth=2, linestyle='--')
                if len(val_forecast) > 0:
                    valid_val_mask = ~np.isnan(val_forecast)
                    ax3.plot(val_idx[valid_val_mask], val_forecast[valid_val_mask],
                             label='Validation Predictions', color='#9467bd', linewidth=2, linestyle='--')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Value')
                ax3.set_title(
                    f'{model_name} Actual vs Predicted (Rolling Forecast)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax5 = fig.add_subplot(gs[2, 0])
                ax5.scatter(y_val.flatten(), val_pred.flatten(),
                            alpha=0.5, color='#17becf')
                min_val = min(y_val.min(), val_pred.min())
                max_val = max(y_val.max(), val_pred.max())
                ax5.plot([min_val, max_val], [
                         min_val, max_val], 'r--', linewidth=2)
                ax5.set_xlabel('Actual Values')
                ax5.set_ylabel('Predicted Values')
                ax5.set_title('Validation Predictions vs Actual (Scatter)')
                ax5.grid(True, alpha=0.3)
                ax_resid = fig.add_subplot(gs[2, 1])
                residuals = y_val.flatten() - val_pred.flatten()
                ax_resid.scatter(val_pred.flatten(), residuals,
                                 alpha=0.5, color='#bcbd22')
                ax_resid.axhline(0, color='r', linestyle='--', linewidth=2)
                ax_resid.set_xlabel('Predicted Values')
                ax_resid.set_ylabel('Residuals')
                ax_resid.set_title('Residuals vs Predicted')
                ax_resid.grid(True, alpha=0.3)
                ax_hist = fig.add_subplot(gs[2, 2])
                sns.histplot(residuals, kde=True, ax=ax_hist, color='#1f77b4')
                ax_hist.set_title('Distribution of Residuals')
                ax_hist.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            except ImportError:
                logger.warning(
                    "Matplotlib or seaborn not available for plotting")
            except Exception as e:
                logger.error(f"Error creating plots: {e}")

        logger.info("="*50)

    def predict(self, test_data, plot_stacks=True, figsize=(20, 12)):
        logger.info("="*50)
        logger.info("FORECASTING")
        logger.info("="*50)
        backcast_len = self.model.backcast_length
        horizon = self.model.forecast_length
        try:
            X_test, y_test = self._create_windows(
                test_data, n=backcast_len // horizon, forecast_len=horizon)
            X_test = torch.from_numpy(X_test).float().to(self.device)
            if y_test is not None:
                y_test = torch.from_numpy(y_test).float().to(self.device)
        except Exception as e:
            logger.error(f"Error preparing test data: {e}")
            return None, None

        self.model.eval()
        with torch.no_grad():
            if self.normalization_type == 'revin':
                predictions = self.model(X_test, mode='norm', cas='fore')
            elif self.normalization_type == 'local':
                X_test_norm, _, test_stats = self._local_normalize(
                    X_test, y_test)  # Dummy y for normalization
                X_test_tensor = torch.from_numpy(
                    X_test_norm).float().to(self.device)
                predictions = self.model(X_test_tensor)
                _, predictions = self._local_denormalize(
                    X_test_tensor, test_stats, predictions)
            elif self.normalization_type == 'global':
                X_test_norm, _, test_stats = self._global_normalize(
                    X_test, np.zeros_like(X_test))
                X_test_tensor = torch.from_numpy(
                    X_test_norm).float().to(self.device)
                predictions = self.model(X_test_tensor)
                predictions = self._global_denormalize(predictions, test_stats)
            else:
                predictions = self.model(X_test)

            stack_contributions = []
            if self.normalization_type == 'revin':
                x_norm, stats = self.model.revin(X_test, "norm")
                x_norm = x_norm.squeeze(-1)
            else:
                x_norm = X_test

            residual = x_norm
            for i, stack in enumerate(self.model.stacks):
                fore, residual = stack(residual)
                if self.normalization_type == 'revin':
                    fore_denorm = self.model.revin((fore, stats), "denorm")
                    stack_contributions.append(fore_denorm.cpu().numpy())
                elif self.normalization_type == 'local':
                    _, fore_denorm = self._local_denormalize(
                        fore, test_stats, fore)
                    stack_contributions.append(fore_denorm)
                elif self.normalization_type == 'global':
                    fore_denorm = self._global_denormalize(fore, test_stats)
                    stack_contributions.append(fore_denorm)
                else:
                    stack_contributions.append(fore.cpu().numpy())
        if isinstance(predictions, torch.Tensor):
            predictions_np = predictions.cpu().numpy()
        else:
            predictions_np = predictions
        test_series = pd.Series(test_data) if not hasattr(
            test_data, 'index') else test_data
        forecast_latest, actual, verif_index = self.build_rolling_from_series(
            predictions, test_series, backcast_len)
        if y_test is not None:
            y_test_np = y_test.cpu().numpy()
            metrics = {
                'MAE': mean_absolute_error(y_test_np.flatten(), predictions_np.flatten()),
                'MSE': mean_squared_error(y_test_np.flatten(), predictions_np.flatten()),
                'RMSE': np.sqrt(mean_squared_error(y_test_np.flatten(), predictions_np.flatten())),
                'MAPE': np.mean(np.abs((y_test_np.flatten() - predictions_np.flatten()) / (y_test_np.flatten() + 1e-8))) * 100
            }
            logger.info("\nTEST PREDICTION METRICS")
            logger.info("-" * 30)
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.2f}%") if 'MAPE' in metric else logger.info(
                    f"{metric}: {value:.4f}")
        if plot_stacks:
            try:
                sns.set_style("whitegrid")
                plt.style.use('seaborn-v0_8-bright')
                fig = plt.figure(figsize=figsize)
                gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
                model_name = self.model.__class__.__name__
                ax1 = fig.add_subplot(gs[0])
                time_steps = np.arange(len(test_data))
                ax1.plot(time_steps, test_data, label='Actual',
                         color='#2ca02c', linewidth=2)
                if len(forecast_latest) > 0:
                    forecast_idx = np.arange(
                        backcast_len, backcast_len + len(forecast_latest))
                    valid_mask = ~np.isnan(forecast_latest)
                    ax1.plot(forecast_idx[valid_mask], forecast_latest[valid_mask],
                             label='Forecast', color='#d62728', linewidth=2, linestyle='--')
                    if valid_mask.any():
                        last_actual_idx = backcast_len - 1
                        first_forecast_idx = forecast_idx[valid_mask][0]
                        if first_forecast_idx == backcast_len:
                            last_actual_value = test_data[last_actual_idx]
                            first_forecast_value = forecast_latest[valid_mask][0]
                            ax1.plot([last_actual_idx, first_forecast_idx],
                                     [last_actual_value, first_forecast_value],
                                     color='#d62728', linewidth=2, linestyle='--')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Value')
                ax1.set_title(f'{model_name} Forecast vs Actual')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax2 = fig.add_subplot(gs[1])
                n_stacks = len(stack_contributions)
                colors = plt.cm.Set3(np.linspace(0, 1, n_stacks))
                stack_means = [np.mean(np.abs(contrib))
                               for contrib in stack_contributions]
                stack_names = [f'Stack {i+1}' for i in range(n_stacks)]
                bars = ax2.bar(stack_names, stack_means,
                               color=colors, alpha=0.7, edgecolor='black')
                ax2.set_ylabel('Average Absolute Contribution')
                ax2.set_title('Stack Contributions to Forecast')
                ax2.grid(True, alpha=0.3, axis='y')
                for bar, value in zip(bars, stack_means):
                    ax2.text(bar.get_x() + bar.get_width()/2., value +
                             value*0.01, f'{value:.3f}', ha='center', va='bottom')
                plt.tight_layout()
                plt.show()
                if len(stack_contributions) > 0:
                    fig2, ax3 = plt.subplots(figsize=(15, 8))
                    n_samples = min(3, predictions_np.shape[0])
                    x_pos = np.arange(horizon)
                    for sample_idx in range(n_samples):
                        bottom = np.zeros(horizon)
                        for stack_idx, contrib in enumerate(stack_contributions):
                            values = contrib[sample_idx]
                            ax3.bar(x_pos + sample_idx * (horizon + 1), values,
                                    bottom=bottom, label=f'Stack {stack_idx+1}' if sample_idx == 0 else "",
                                    color=colors[stack_idx], alpha=0.7, width=0.8)
                            bottom += values
                        ax3.plot(x_pos + sample_idx * (horizon + 1), predictions_np[sample_idx],
                                 'ko-', markersize=6, linewidth=2,
                                 label='Total Forecast' if sample_idx == 0 else "")
                    ax3.set_xlabel('Forecast Horizon')
                    ax3.set_ylabel('Value')
                    ax3.set_title(
                        'Stack Contributions by Forecast Horizon (Sample Windows)')
                    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax3.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                logger.error(f"Error creating forecast plots: {e}")
        logger.info("="*50)
        return predictions_np, stack_contributions
