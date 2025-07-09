import numpy as np
import matplotlib.pyplot as plt

class NBeatsBlock:
    def __init__(self, input_size, hidden_size, theta_size, forecast_horizon, basis_type='chebyshev', degree=3, periodicity=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.theta_size = theta_size
        self.forecast_horizon = forecast_horizon
        self.basis_type = basis_type
        self.degree = degree
        self.periodicity = periodicity
        # Initialize weights and biases with proper scaling
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(hidden_size)
        self.W4 = np.random.randn(hidden_size, theta_size) * np.sqrt(2.0 / hidden_size)
        self.b4 = np.zeros(theta_size)
        # Initialize basis functions
        if basis_type == 'chebyshev':
            self.forecast_basis = self._chebyshev_basis(forecast_horizon, degree)
            self.backcast_basis = self._chebyshev_basis(input_size, degree)
        elif basis_type == 'fourier':
            if periodicity is None:
                raise ValueError("Periodicity required for Fourier basis.")
            self.forecast_basis = self._fourier_basis(forecast_horizon, degree, periodicity)
            self.backcast_basis = self._fourier_basis(input_size, degree, periodicity)
        elif basis_type == 'generic':
            self.forecast_basis = np.random.randn(forecast_horizon, theta_size) * 0.01
            self.backcast_basis = np.random.randn(input_size, theta_size) * 0.01
        else:
            raise ValueError("Invalid basis_type. Choose from 'chebyshev', 'fourier', or 'generic'.")
        # Store gradients
        self.gradients = {}
    def _chebyshev_basis(self, length, degree):
        """Generate Chebyshev polynomial basis functions"""
        t = np.linspace(-1, 1, length)
        basis = np.zeros((length, degree + 1))
        for i in range(length):
            basis[i, 0] = 1.0
            if degree >= 1:
                basis[i, 1] = t[i]
            for n in range(2, degree + 1):
                basis[i, n] = 2 * t[i] * basis[i, n-1] - basis[i, n-2]
        return basis
    def _fourier_basis(self, length, num_terms, periodicity):
        """Generate Fourier basis functions"""
        t = np.arange(length)
        basis = np.zeros((length, 2 * num_terms + 1))
        basis[:, 0] = 1.0  # DC component
        for n in range(1, num_terms + 1):
            basis[:, 2 * n - 1] = np.cos(2 * np.pi * n * t / periodicity)
            basis[:, 2 * n] = np.sin(2 * np.pi * n * t / periodicity)
        return basis
    def _relu(self, x):
        return np.maximum(0, x)
    def _relu_deriv(self, x):
        return np.where(x > 0, 1.0, 0.0)
    def forward(self, x):
        """Forward pass through the block"""
        self.x = x
        # Layer 1
        self.z1 = np.dot(x, self.W1) + self.b1
        self.h1 = self._relu(self.z1)
        # Layer 2
        self.z2 = np.dot(self.h1, self.W2) + self.b2
        self.h2 = self._relu(self.z2)
        # Layer 3
        self.z3 = np.dot(self.h2, self.W3) + self.b3
        self.h3 = self._relu(self.z3)
        # Output layer (theta parameters)
        self.theta = np.dot(self.h3, self.W4) + self.b4
        # Generate forecast and backcast using basis functions
        forecast = np.dot(self.forecast_basis, self.theta)
        backcast = np.dot(self.backcast_basis, self.theta)
        self.forecast = forecast
        self.backcast = backcast
        return forecast, backcast
    def backward(self, d_forecast, d_backcast):
        """Backward pass to compute gradients"""
        # Gradient w.r.t. theta
        d_theta = np.dot(self.forecast_basis.T, d_forecast) + np.dot(self.backcast_basis.T, d_backcast)
        # Gradients for output layer
        self.gradients['W4'] = np.outer(self.h3, d_theta)
        self.gradients['b4'] = d_theta.copy()
        # Gradient w.r.t. h3
        d_h3 = np.dot(d_theta, self.W4.T)
        # Gradient through ReLU activation
        d_z3 = d_h3 * self._relu_deriv(self.z3)
        # Gradients for layer 3
        self.gradients['W3'] = np.outer(self.h2, d_z3)
        self.gradients['b3'] = d_z3.copy()
        # Gradient w.r.t. h2
        d_h2 = np.dot(d_z3, self.W3.T)
        # Gradient through ReLU activation
        d_z2 = d_h2 * self._relu_deriv(self.z2)
        # Gradients for layer 2
        self.gradients['W2'] = np.outer(self.h1, d_z2)
        self.gradients['b2'] = d_z2.copy()
        # Gradient w.r.t. h1
        d_h1 = np.dot(d_z2, self.W2.T)
        # Gradient through ReLU activation
        d_z1 = d_h1 * self._relu_deriv(self.z1)
        # Gradients for layer 1
        self.gradients['W1'] = np.outer(self.x, d_z1)
        self.gradients['b1'] = d_z1.copy()
        # Gradient w.r.t. input
        d_input = np.dot(d_z1, self.W1.T)
        # Update generic basis functions if applicable
        if self.basis_type == 'generic':
            self.gradients['forecast_basis'] = np.outer(d_forecast, self.theta)
            self.gradients['backcast_basis'] = np.outer(d_backcast, self.theta)
        return d_input
class NBeatsStack:
    def __init__(self, input_size, hidden_size, theta_size, forecast_horizon, num_blocks, basis_type='chebyshev', degree=3, periodicity=None):
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.blocks = [
            NBeatsBlock(input_size, hidden_size, theta_size, forecast_horizon, basis_type, degree, periodicity)
            for _ in range(num_blocks)
        ]
    def forward(self, x):
        """Forward pass through all blocks in the stack"""
        residual = x.copy()
        stack_forecast = np.zeros(self.forecast_horizon)
        self.residuals = [residual.copy()]
        self.forecasts = []
        for block in self.blocks:
            forecast, backcast = block.forward(residual)
            stack_forecast += forecast
            self.forecasts.append(forecast)
            residual = residual - backcast  # Update residual
            self.residuals.append(residual.copy())
        return stack_forecast, residual
    def backward(self, d_forecast, d_residual):
        """Backward pass through all blocks in the stack"""
        d_input = d_residual.copy()
        # Backpropagate through blocks in reverse order
        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            # Each block contributes to the stack forecast
            d_block_forecast = d_forecast.copy()
            d_block_backcast = -d_input  # Negative because residual = input - backcast
            d_input = block.backward(d_block_forecast, d_block_backcast)
        return d_input
class NBeatsModel:
    def __init__(self, input_size, hidden_size, forecast_horizon, stack_configs):
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.stacks = [
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
        ]
    def forward(self, x):
        """Forward pass through all stacks"""
        residual = x.copy()
        total_forecast = np.zeros(self.forecast_horizon)
        self.stack_forecasts = []
        self.residuals = [residual.copy()]
        for stack in self.stacks:
            stack_forecast, stack_residual = stack.forward(residual)
            total_forecast += stack_forecast
            self.stack_forecasts.append(stack_forecast)
            residual = stack_residual  # Pass residual to next stack
            self.residuals.append(residual.copy())
        return total_forecast, residual
    def backward(self, d_forecast, d_residual):
        """Backward pass through all stacks"""
        d_input = d_residual.copy()
        # Backpropagate through stacks in reverse order
        for i in range(len(self.stacks) - 1, -1, -1):
            stack = self.stacks[i]
            # Each stack contributes to the total forecast
            d_stack_forecast = d_forecast.copy()
            d_stack_residual = d_input
            d_input = stack.backward(d_stack_forecast, d_stack_residual)
        return d_input
    def get_parameters(self):
        """Get all model parameters"""
        parameters = []
        for stack_idx, stack in enumerate(self.stacks):
            for block_idx, block in enumerate(stack.blocks):
                param_prefix = f"stack_{stack_idx}_block_{block_idx}"
                parameters.extend([
                    (f"{param_prefix}_W1", block.W1),
                    (f"{param_prefix}_b1", block.b1),
                    (f"{param_prefix}_W2", block.W2),
                    (f"{param_prefix}_b2", block.b2),
                    (f"{param_prefix}_W3", block.W3),
                    (f"{param_prefix}_b3", block.b3),
                    (f"{param_prefix}_W4", block.W4),
                    (f"{param_prefix}_b4", block.b4)
                ])
                if block.basis_type == 'generic':
                    parameters.extend([
                        (f"{param_prefix}_forecast_basis", block.forecast_basis),
                        (f"{param_prefix}_backcast_basis", block.backcast_basis)
                    ])
        return parameters
    def get_gradients(self):
        """Get all gradients"""
        gradients = {}
        for stack_idx, stack in enumerate(self.stacks):
            for block_idx, block in enumerate(stack.blocks):
                param_prefix = f"stack_{stack_idx}_block_{block_idx}"
                for param_name, grad in block.gradients.items():
                    gradients[f"{param_prefix}_{param_name}"] = grad
        return gradients
class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    def step(self, parameters, gradients):
        """Update parameters using Adam optimization"""
        self.t += 1
        for param_name, param in parameters:
            if param_name not in gradients:
                continue
            grad = gradients[param_name]
            # Initialize momentum terms if needed
            if param_name not in self.m:
                self.m[param_name] = np.zeros_like(param)
                self.v[param_name] = np.zeros_like(param)
            # Update momentum terms
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
            # Bias correction
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            # Update parameter
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
def generate_synthetic_data(length, period=10):
    """Generate synthetic time series data with trend, seasonality, and noise"""
    t = np.linspace(0, length / period, length)
    trend = 0.01 * t + 0.001 * t**2
    seasonality = np.sin(2 * np.pi * t) + 0.5 * np.cos(4 * np.pi * t)
    noise = np.random.randn(length) * 0.1
    return trend + seasonality + noise
def create_sliding_windows(data, input_size, forecast_horizon):
    """Create sliding window samples from time series data"""
    X, y = [], []
    for i in range(len(data) - input_size - forecast_horizon + 1):
        X.append(data[i:i + input_size])
        y.append(data[i + input_size:i + input_size + forecast_horizon])
    return np.array(X), np.array(y)
def normalize_data(X, y):
    """Normalize data to zero mean and unit variance"""
    # Compute statistics from training data only
    X_flat = X.flatten()
    mean = np.mean(X_flat)
    std = np.std(X_flat)
    # Ensure std is not too small
    if std < 1e-6:
        std = 1.0
    X_norm = (X - mean) / std
    y_norm = (y - mean) / std
    return X_norm, y_norm, mean, std
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    # Flatten arrays to handle multi-dimensional predictions
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    # Calculate SMAPE with proper handling of zero values
    denominator = np.abs(y_true_flat) + np.abs(y_pred_flat) + 1e-8
    smape_values = 2 * np.abs(y_pred_flat - y_true_flat) / denominator
    # Remove any infinite or NaN values
    smape_values = smape_values[np.isfinite(smape_values)]
    return 100 * np.mean(smape_values)
def mae(y_true, y_pred):
    """Mean Absolute Error"""
    # Flatten arrays to handle multi-dimensional predictions
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return np.mean(np.abs(y_pred_flat - y_true_flat))
def train_nbeats(model, X, y, epochs, batch_size, learning_rate=0.001, backcast_loss_weight=0.1):
    """Train the N-BEATS model"""
    optimizer = AdamOptimizer(lr=learning_rate)
    losses = []
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X))
        epoch_loss = 0
        num_batches = 0
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X, batch_y = X[batch_indices], y[batch_indices]
            batch_loss = 0
            # Accumulate gradients over batch
            accumulated_gradients = {}
            parameters = model.get_parameters()
            for x, y_true in zip(batch_X, batch_y):
                # Forward pass
                y_pred, residual = model.forward(x)
                # Compute losses
                forecast_loss = np.mean((y_pred - y_true) ** 2)
                backcast_loss = np.mean(residual ** 2)  # Residual should be small
                total_loss = forecast_loss + backcast_loss_weight * backcast_loss
                batch_loss += total_loss
                # Backward pass
                d_forecast = 2 * (y_pred - y_true) / len(y_pred)
                d_residual = 2 * backcast_loss_weight * residual / len(residual)
                model.backward(d_forecast, d_residual)
                # Accumulate gradients
                gradients = model.get_gradients()
                for param_name, grad in gradients.items():
                    if param_name in accumulated_gradients:
                        accumulated_gradients[param_name] += grad
                    else:
                        accumulated_gradients[param_name] = grad.copy()
            # Average gradients over batch
            for param_name in accumulated_gradients:
                accumulated_gradients[param_name] /= len(batch_X)
            # Gradient clipping to prevent exploding gradients
            max_grad_norm = 1.0
            total_norm = 0
            for param_name in accumulated_gradients:
                total_norm += np.sum(accumulated_gradients[param_name] ** 2)
            total_norm = np.sqrt(total_norm)
            if total_norm > max_grad_norm:
                for param_name in accumulated_gradients:
                    accumulated_gradients[param_name] *= max_grad_norm / total_norm
            # Update parameters
            optimizer.step(parameters, accumulated_gradients)
            batch_loss /= len(batch_X)
            epoch_loss += batch_loss
            num_batches += 1
        epoch_loss /= num_batches
        losses.append(epoch_loss)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")
    return losses
def visualize_results(model, X, y, mean, std):
    """Visualize model predictions"""
    # Take first test sample
    x_sample = X[0]
    y_true = y[0]
    # Make prediction
    y_pred, residual = model.forward(x_sample)
    # Denormalize
    x_sample = x_sample * std + mean
    y_true = y_true * std + mean
    y_pred = y_pred * std + mean
    residual = residual * std + mean
    plt.figure(figsize=(15, 10))
    # Plot 1: Input, forecast, and residual
    plt.subplot(3, 1, 1)
    plt.plot(range(len(x_sample)), x_sample, 'b-', label='Input', linewidth=2)
    plt.plot(range(len(x_sample), len(x_sample) + len(y_true)), y_true, 'g-', label='True Forecast', linewidth=2)
    plt.plot(range(len(x_sample), len(x_sample) + len(y_pred)), y_pred, 'r--', label='Predicted Forecast', linewidth=2)
    plt.plot(range(len(residual)), residual, 'm:', label='Final Residual', alpha=0.7)
    plt.axvline(x=len(x_sample), color='k', linestyle=':', alpha=0.5)
    plt.legend()
    plt.title('N-BEATS Prediction Results')
    plt.grid(True, alpha=0.3)
    # Plot 2: Stack-specific forecasts
    plt.subplot(3, 1, 2)
    for i, stack_forecast in enumerate(model.stack_forecasts):
        stack_forecast_denorm = stack_forecast * std + mean
        plt.plot(range(len(x_sample), len(x_sample) + len(stack_forecast_denorm)), 
                stack_forecast_denorm, label=f'Stack {i+1} Forecast', linewidth=1.5)
    plt.plot(range(len(x_sample), len(x_sample) + len(y_pred)), y_pred, 'k-', 
             label='Total Forecast', linewidth=2, alpha=0.8)
    plt.axvline(x=len(x_sample), color='k', linestyle=':', alpha=0.5)
    plt.legend()
    plt.title('Stack Decomposition')
    plt.grid(True, alpha=0.3)
    # Plot 3: Residuals evolution
    plt.subplot(3, 1, 3)
    for i, res in enumerate(model.residuals):
        if i < len(model.residuals) - 1:  # Don't plot the final residual twice
            res_denorm = res * std + mean
            plt.plot(range(len(res_denorm)), res_denorm, alpha=0.7, label=f'After Stack {i}' if i > 0 else 'Initial Input')
    plt.legend()
    plt.title('Residual Evolution Through Stacks')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()