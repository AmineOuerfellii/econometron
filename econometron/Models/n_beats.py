import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class Generic_Basis(nn.Module):
    def __init__(self, backcast_length,forecast_length):
        super(Generic_Basis, self).__init__()
        #y_Hat_{l}=V_{l}**f * theta_{l}**f
        #x_Hat_{l}=V_{l}**b * theta_{l}**b
        self.backcast_length=backcast_length
        self.forecast_length=forecast_length
        #V matrix
        self.V_f = nn.Parameter(torch.randn(forecast_length,forecast_length) * 0.01)
        self.V_b= nn.Parameter(torch.randn(backcast_length,backcast_length) * 0.01)
        # Bias terms
        self.b_f = nn.Parameter(torch.zeros(forecast_length))
        self.b_b = nn.Parameter(torch.zeros(backcast_length))
        ##
        def forward(self,theta_b,theta_f):
          forecast=torch.matmul(self.V_f,theta_f)+self.b_f
          backcast=torch.matmul(self.V_b,theta_b)+self.b_b
          return forecast,backcast

class ChebyshevBasis(nn.Module):
    def __init__(self, backcast_length, forecast_length, degree=3):
        super(ChebyshevBasis, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.degree = degree
        t_back = np.linspace(-1, 1, backcast_length)
        t_fore = np.linspace(-1, 1, forecast_length)
        basis_back = np.zeros((backcast_length, degree + 1))
        basis_fore = np.zeros((forecast_length, degree + 1))
        for i in range(backcast_length):
            basis_back[i, 0] = 1.0
            if degree >= 1:
                basis_back[i, 1] = t_back[i]
            for n in range(2, degree + 1):
                basis_back[i, n] = 2 * t_back[i] * basis_back[i, n-1] - basis_back[i, n-2]
        for i in range(forecast_length):
            basis_fore[i, 0] = 1.0
            if degree >= 1:
                basis_fore[i, 1] = t_fore[i]
            for n in range(2, degree + 1):
                basis_fore[i, n] = 2 * t_fore[i] * basis_fore[i, n-1] - basis_fore[i, n-2]
        self.register_buffer('forecast_basis', torch.tensor(basis_fore, dtype=torch.float32))
        self.register_buffer('backcast_basis', torch.tensor(basis_back, dtype=torch.float32))

    def forward(self, theta):
        forecast = torch.matmul(self.forecast_basis, theta.unsqueeze(-1)).squeeze(-1)
        backcast = torch.matmul(self.backcast_basis, theta.unsqueeze(-1)).squeeze(-1)
        return backcast, forecast

class FourierBasis(nn.Module):
    def __init__(self, backcast_length, forecast_length, num_terms, periodicity):
        super(FourierBasis, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        t_back = np.arange(backcast_length)
        t_fore = np.arange(forecast_length)
        basis_back = np.zeros((backcast_length, 2 * num_terms + 1))
        basis_fore = np.zeros((forecast_length, 2 * num_terms + 1))
        basis_back[:, 0] = 1.0
        basis_fore[:, 0] = 1.0
        for n in range(1, num_terms + 1):
            basis_back[:, 2 * n - 1] = np.cos(2 * np.pi * n * t_back / periodicity)
            basis_back[:, 2 * n] = np.sin(2 * np.pi * n * t_back / periodicity)
            basis_fore[:, 2 * n - 1] = np.cos(2 * np.pi * n * t_fore / periodicity)
            basis_fore[:, 2 * n] = np.sin(2 * np.pi * n * t_fore / periodicity)
        self.register_buffer('forecast_basis', torch.tensor(basis_fore, dtype=torch.float32))
        self.register_buffer('backcast_basis', torch.tensor(basis_back, dtype=torch.float32))

    def forward(self, theta):
        forecast = torch.matmul(self.forecast_basis, theta.unsqueeze(-1)).squeeze(-1)
        backcast = torch.matmul(self.backcast_basis, theta.unsqueeze(-1)).squeeze(-1)
        return backcast, forecast

class PolynomialBasis(nn.Module):
    def __init__(self, backcast_length, forecast_length, degree=3):
        super(PolynomialBasis, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.degree = degree
        t_back = np.linspace(0, 1, backcast_length)
        t_fore = np.linspace(1, 1 + forecast_length/backcast_length, forecast_length)
        basis_back = np.zeros((backcast_length, degree + 1))
        basis_fore = np.zeros((forecast_length, degree + 1))
        for d in range(degree + 1):
            basis_back[:, d] = t_back ** d
            basis_fore[:, d] = t_fore ** d
        self.register_buffer('forecast_basis', torch.tensor(basis_fore, dtype=torch.float32))
        self.register_buffer('backcast_basis', torch.tensor(basis_back, dtype=torch.float32))

    def forward(self, theta):
        forecast = torch.matmul(self.forecast_basis, theta.unsqueeze(-1)).squeeze(-1)
        backcast = torch.matmul(self.backcast_basis, theta.unsqueeze(-1)).squeeze(-1)
        return backcast, forecast

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_size, theta_size, forecast_horizon, basis_type='chebyshev', degree=3, periodicity=None):
        super(NBeatsBlock, self).__init__()
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.basis_type = basis_type
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, theta_size)
        )
        if basis_type == 'chebyshev':
            self.basis_function = ChebyshevBasis(input_size, forecast_horizon, degree)
        elif basis_type == 'fourier':
            if periodicity is None:
                raise ValueError("Periodicity required for Fourier basis.")
            self.basis_function = FourierBasis(input_size, forecast_horizon, degree, periodicity)
        elif basis_type == 'generic':
            self.basis_function = GenericBasis(input_size, forecast_horizon)
        elif basis_type == 'polynomial':
            self.basis_function = PolynomialBasis(input_size, forecast_horizon, degree)
        else:
            raise ValueError("Invalid basis_type. Choose from 'chebyshev', 'fourier', 'generic', or 'polynomial'.")
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        theta = self.fc_stack(x)
        backcast, forecast = self.basis_function(theta)
        return forecast, backcast

class NBeatsStack(nn.Module):
    def __init__(self, input_size, hidden_size, theta_size, forecast_horizon, num_blocks, basis_type='chebyshev', degree=3, periodicity=None):
        super(NBeatsStack, self).__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, hidden_size, theta_size, forecast_horizon, basis_type, degree, periodicity)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        residual = x
        stack_forecast = torch.zeros(x.shape[0], self.blocks[0].forecast_horizon, device=x.device)
        forecasts = []
        residuals = [residual]

        for block in self.blocks:
            forecast, backcast = block(residual)
            stack_forecast += forecast
            forecasts.append(forecast)
            residual = residual - backcast
            residuals.append(residual)

        return stack_forecast, residual, forecasts, residuals

class NBeatsModel(nn.Module):
    def __init__(self, input_size, hidden_size, forecast_horizon, stack_configs):
        super(NBeatsModel, self).__init__()
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.stacks = nn.ModuleList([
            NBeatsStack(
                input_size=input_size,
                hidden_size=hidden_size,
                theta_size=config['theta_size'],
                forecast_horizon=forecast_horizon,
                num_blocks=config['num_blocks'],
                basis_type=config['basis_type'],
                degree=config['degree'],
                periodicity=config.get('periodicity')
            )
            for config in stack_configs
        ])

    def forward(self, x):
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Input shape {x.shape[-1]} does not match expected input_size {self.input_size}")
        residual = x
        total_forecast = torch.zeros(x.shape[0], self.forecast_horizon, device=x.device)
        stack_forecasts = []
        residuals = [residual]

        for stack in self.stacks:
            stack_forecast, stack_residual, block_forecasts, block_residuals = stack(residual)
            total_forecast += stack_forecast
            stack_forecasts.append(stack_forecast)
            residual = stack_residual
            residuals.append(residual)

        return total_forecast, residual, stack_forecasts, residuals

    def train_model(self, X, y, epochs=100, batch_size=32, learning_rate=0.001, backcast_loss_weight=0.1, validation_data=None):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                forecast, residual, _, _ = self(batch_x)
                forecast_loss = torch.mean((forecast - batch_y) ** 2)
                backcast_loss = torch.mean(residual ** 2)
                loss = forecast_loss + backcast_loss_weight * backcast_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if validation_data is not None:
                val_X, val_y = validation_data
                val_loss = self._validate(val_X, val_y, backcast_loss_weight)
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
            elif (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}")

        return losses

    def _validate(self, X, y, backcast_loss_weight):
        self.eval()
        with torch.no_grad():
            X, y = torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)
            forecast, residual, _, _ = self(X)
            forecast_loss = torch.mean((forecast - y) ** 2)
            backcast_loss = torch.mean(residual ** 2)
            return (forecast_loss + backcast_loss_weight * backcast_loss).item()

    def fit(self, X, y, epochs=100, batch_size=32, learning_rate=0.001, backcast_loss_weight=0.1, validation_data=None):
        return self.train_model(X, y, epochs, batch_size, learning_rate, backcast_loss_weight, validation_data)

    def forecast(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            forecast, _, stack_forecasts, _ = self(X)
            return forecast.cpu().numpy(), stack_forecasts

    def stats(self):
        return {
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'training_loss': self.history['loss'][-1] if hasattr(self, 'history') and self.history['loss'] else None,
            'num_stacks': len(self.stacks),
            'input_size': self.input_size,
            'forecast_horizon': self.forecast_horizon,
            'stack_configs': [
                {
                    'basis_type': stack.blocks[0].basis_type,
                    'num_blocks': len(stack.blocks),
                    'theta_size': stack.blocks[0].fc_stack[-2].out_features
                } for stack in self.stacks
            ]
        }

    def visualize(self, X, y, mean, std):
        forecast, stack_forecasts = self.forecast(X)
        x_sample = X[0] * std + mean
        y_true = y[0] * std + mean
        y_pred = forecast[0] * std + mean
        stack_forecasts_denorm = [sf[0].cpu().numpy() * std + mean for sf in stack_forecasts]

        plt.figure(figsize=(15, 10))

        # Plot 1: Input and forecast
        plt.subplot(2, 1, 1)
        plt.plot(range(len(x_sample)), x_sample, 'b-', label='Input', linewidth=2)
        plt.plot(range(len(x_sample), len(x_sample) + len(y_true)), y_true, 'g-', label='True Forecast', linewidth=2)
        plt.plot(range(len(x_sample), len(x_sample) + len(y_pred)), y_pred, 'r--', label='Predicted Forecast', linewidth=2)
        plt.axvline(x=len(x_sample), color='k', linestyle=':', alpha=0.5)
        plt.legend()
        plt.title('N-BEATS Prediction Results')
        plt.grid(True, alpha=0.3)

        # Plot 2: Stack-specific forecasts
        plt.subplot(2, 1, 2)
        for i, stack_forecast in enumerate(stack_forecasts_denorm):
            plt.plot(range(len(x_sample), len(x_sample) + len(stack_forecast)),
                     stack_forecast, label=f'Stack {i+1} ({self.stacks[i].blocks[0].basis_type})', linewidth=1.5)
        plt.plot(range(len(x_sample), len(x_sample) + len(y_pred)), y_pred, 'k-',
                 label='Total Forecast', linewidth=2, alpha=0.8)
        plt.axvline(x=len(x_sample), color='k', linestyle=':', alpha=0.5)
        plt.legend()
        plt.title('Stack Decomposition')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @property
    def device(self):
        return next(self.parameters()).device

def create_sliding_windows(data, input_size, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - input_size - forecast_horizon + 1):
        X.append(data[i:i + input_size])
        y.append(data[i + input_size:i + input_size + forecast_horizon])
    return np.array(X), np.array(y)

def normalize_data(X, y):
    X_flat = X.flatten()
    mean = np.mean(X_flat)
    std = np.std(X_flat) if np.std(X_flat) > 1e-6 else 1.0
    X_norm = (X - mean) / std
    y_norm = (y - mean) / std
    return X_norm, y_norm, mean, std

def smape(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    denominator = np.abs(y_true_flat) + np.abs(y_pred_flat) + 1e-8
    smape_values = 2 * np.abs(y_pred_flat - y_true_flat) / denominator
    smape_values = smape_values[np.isfinite(smape_values)]
    return 100 * np.mean(smape_values)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred.flatten() - y_true.flatten()))