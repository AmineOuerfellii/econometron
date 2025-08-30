import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch
from econometron.Models.Neuralnets import Trainer_ts, NBEATS

# Fixture to create a sample dataset
@pytest.fixture
def sample_data():
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    data = np.sin(0.1 * t) + 0.1 * np.random.randn(1000)
    return data

# Fixture to create a basic NBEATS model
@pytest.fixture
def nbeats_model():
    stack_configs = [
        {'num_B_per_S': 2, 'Blocks': ['G', 'S'], 'Harmonics': [4], 'Degree': [2],
         'Dropout': [0.1], 'Layer_size': [64], 'num_lay_per_B': [2], 'share_weights': False}
    ]
    model = NBEATS(n=2, h=10, n_s=1, stack_configs=stack_configs)
    return model

# Fixture to create a Trainer_ts instance
@pytest.fixture
def trainer(nbeats_model):
    return Trainer_ts(model=nbeats_model, normalization_type='revin', device='cpu', Seed=42)

# Test initialization
def test_trainer_initialization(nbeats_model):
    trainer = Trainer_ts(model=nbeats_model, normalization_type='revin', device='cpu', Seed=42)
    assert isinstance(trainer.model, NBEATS)
    assert trainer.normalization_type == 'revin'
    assert trainer.device == 'cpu'
    assert trainer.seed == 42
    assert torch.equal(trainer.model.state_dict()['stacks.0.blocks.0.FC_stack.0.weight'],
                       trainer.model_initial_state['stacks.0.blocks.0.FC_stack.0.weight'])

def test_trainer_invalid_model():
    with pytest.raises(ValueError, match="Model must be a PyTorch nn.Module instance"):
        Trainer_ts(model="not_a_model", normalization_type='revin')

def test_trainer_invalid_normalization(nbeats_model):
    trainer = Trainer_ts(model=nbeats_model, normalization_type='invalid', device='cpu')
    assert trainer.normalization_type is None

# Test data validation
def test_validate_data_numpy(trainer, sample_data):
    validated_data = trainer._validate_data(sample_data)
    assert isinstance(validated_data, np.ndarray)
    assert validated_data.dtype == np.float32
    assert len(validated_data) == len(sample_data)
    assert not np.isnan(validated_data).any()

def test_validate_data_pandas(trainer):
    df = pd.DataFrame({'value': np.random.randn(100)})
    validated_data = trainer._validate_data(df)
    assert isinstance(validated_data, np.ndarray)
    assert validated_data.shape == (100,)
    assert validated_data.dtype == np.float32

def test_validate_data_nan_handling(trainer):
    data = np.array([1, 2, np.nan, 4, 5])
    validated_data = trainer._validate_data(data)
    assert len(validated_data) == 4
    assert not np.isnan(validated_data).any()

def test_validate_data_inf_handling(trainer):
    data = np.array([1, 2, np.inf, 4, 5])
    validated_data = trainer._validate_data(data)
    assert len(validated_data) == 4
    assert not np.isinf(validated_data).any()

# Test window creation
def test_create_windows(trainer, sample_data):
    X, y = trainer._create_windows(sample_data, n=2, forecast_len=10)
    assert X.shape == (971, 20)  
    assert y.shape == (971, 10)
    assert np.allclose(X[0, :], sample_data[:20])
    assert np.allclose(y[0, :], sample_data[20:30])

def test_create_windows_insufficient_data(trainer):
    short_data = np.random.randn(15)
    with pytest.raises(ValueError, match="Data length"):
        trainer._create_windows(short_data, n=2, forecast_len=10)

# Test normalization
def test_local_normalize(trainer):
    X = np.random.randn(10, 20)
    y = np.random.randn(10, 5)
    X_norm, y_norm, stats = trainer._local_normalize(X, y)
    assert X_norm.shape == X.shape
    assert y_norm.shape == y.shape
    assert stats.shape == (10, 4)
    assert np.allclose(X_norm.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(y_norm.mean(axis=1), 0, atol=1e-5)

def test_global_normalize(trainer):
    X = np.random.randn(10, 20)
    y = np.random.randn(10, 5)
    trainer.data = np.random.randn(100)  # Set data for global stats
    X_norm, y_norm, stats = trainer._global_normalize(X, y)
    assert X_norm.shape == X.shape
    assert y_norm.shape == y.shape
    assert 'mean' in stats and 'std' in stats
    assert np.allclose(X_norm * stats['std'] + stats['mean'], X)

# Test training
@patch('matplotlib.pyplot.show')  # Mock plotting to avoid GUI issues
def test_fit_basic(mock_show, trainer, sample_data):
    history = trainer.fit(
        Data=sample_data,
        N=2,
        Horizon=10,
        max_epochs=2,
        batch_size=16,
        optimizer='adam',
        lr=1e-3,
        loss_fun='mae',
        early_stopping=5,
        verbose=False
    )
    assert len(history['train_loss']) == 2
    assert len(history['val_loss']) == 2
    assert len(history['lr']) == 2
    assert len(history['epoch']) == 2
    assert trainer.best_model_state is not None

def test_fit_early_stopping(trainer, sample_data):
    history = trainer.fit(
        Data=sample_data,
        N=2,
        Horizon=10,
        max_epochs=10,
        batch_size=16,
        early_stopping=2,
        verbose=False
    )
    assert len(history['epoch']) <= 10
    assert trainer.best_model_state is not None

# def test_fit_invalid_optimizer(trainer, sample_data):
#     with pytest.raises(ValueError, match="Data length (15) must be at least backcast_length + forecast_length (30)"):
#         trainer.fit(
#             Data=sample_data[:15], 
#             N=2,
#             Horizon=10,
#             optimizer='invalid',
#             verbose=False
#         )
#         assert trainer.optimizer=="adam"

# Test learning rate finder
@patch('matplotlib.pyplot.show')
def test_find_optimal_lr(mock_show, trainer, sample_data):
    lrs, losses, suggested_lr = trainer.find_optimal_lr(
        data=sample_data,
        back_coeff=2,
        Horizon=10,
        batch_size=16,
        start_lr=1e-7,
        end_lr=1e-3,
        num_iter=20,
        plot=False
    )
    assert len(lrs) == len(losses)
    assert suggested_lr >= 1e-7 and suggested_lr <= 1e-3
    assert len(lrs) <= 20

# Test prediction
@patch('matplotlib.pyplot.show')
def test_predict(mock_show, trainer, sample_data):
    trainer.fit(
        Data=sample_data,
        N=2,
        Horizon=10,
        max_epochs=2,
        batch_size=16,
        verbose=False
    )
    test_data = sample_data[-50:]  # Last 50 points
    predictions, stack_contributions = trainer.predict(test_data, plot_stacks=False)
    assert predictions.shape == (21, 10)  
    assert len(stack_contributions) == 1 
    assert stack_contributions[0].shape == (21, 10)

# Test out-of-sample forecasting
@patch('matplotlib.pyplot.show')
def test_forecast_out_of_sample(mock_show, trainer, sample_data):
    trainer.fit(
        Data=sample_data,
        N=2,
        Horizon=10,
        max_epochs=2,
        batch_size=16,
        verbose=False
    )
    forecasts = trainer.forecast_out_of_sample(steps=20, plot=False)
    assert len(forecasts) == 20
    assert isinstance(forecasts, np.ndarray)
    assert not np.isnan(forecasts).any()

def test_forecast_out_of_sample_no_data(trainer):
    with pytest.raises(ValueError, match="No data found"):
        trainer.forecast_out_of_sample(steps=10)

def test_forecast_out_of_sample_insufficient_data(trainer):
    trainer.data = np.random.randn(10)  # Shorter than backcast_length
    with pytest.raises(ValueError, match="Data length"):
        trainer.forecast_out_of_sample(steps=10)

if __name__ == "__main__":
    pytest.main([__file__])