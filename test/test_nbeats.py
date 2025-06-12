from econometron.Models.n_beats import *
import pytest
# Parameters
def test_nbeats():
    input_size = 20
    hidden_size = 32  # Reduced from 64
    forecast_horizon = 5
    epochs = 200  # Increased epochs
    batch_size = 16  # Reduced batch size
    learning_rate = 0.0005  # Reduced learning rate
    backcast_loss_weight = 0.1

    # Simplified stack configurations for better training
    stack_configs = [
        {'num_blocks': 2, 'basis_type': 'chebyshev', 'degree': 2, 'theta_size': 3},  # Trend stack
        {'num_blocks': 2, 'basis_type': 'fourier', 'degree': 1, 'theta_size': 3, 'periodicity': 10.0},  # Seasonality stack
        {'num_blocks': 2, 'basis_type': 'generic', 'degree': 0, 'theta_size': 4}  # Generic stack
    ]

    print("Importing data...")
    import scipy.io
    data = scipy.io.loadmat("Z.mat") # More data
    data=data["Z"]
    data=data[1,:]

    print("Preprocessing data...")
    X, y = create_sliding_windows(data, input_size, forecast_horizon)
    X, y, mean, std = normalize_data(X, y)

    # Split data
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Initialize model
    print("Initializing N-BEATS model...")
    model = NBeatsModel(input_size, hidden_size, forecast_horizon, stack_configs)

    # Train model
    print("Training model...")
    losses = train_nbeats(model, X_train, y_train, epochs, batch_size, learning_rate, backcast_loss_weight)

    # Evaluate on test set
    print("Evaluating model...")
    test_predictions = []
    for x in X_test:
        y_pred, _ = model.forward(x)
        test_predictions.append(y_pred)

    test_predictions = np.array(test_predictions)

    # Denormalize for evaluation
    y_test_denorm = y_test * std + mean
    test_predictions_denorm = test_predictions * std + mean

    # Calculate metrics
    test_smape = smape(y_test_denorm, test_predictions_denorm)
    test_mae = mae(y_test_denorm, test_predictions_denorm)

    # Additional debugging metrics
    print(f"\nTest Results:")
    print(f"sMAPE: {test_smape:.2f}%")
    print(f"MAE: {test_mae:.4f}")
    print(f"Mean of true values: {np.mean(y_test_denorm):.4f}")
    print(f"Mean of predictions: {np.mean(test_predictions_denorm):.4f}")
    print(f"Std of true values: {np.std(y_test_denorm):.4f}")
    print(f"Std of predictions: {np.std(test_predictions_denorm):.4f}")

    # Calculate simple MSE and RMSE for comparison
    mse = np.mean((y_test_denorm - test_predictions_denorm) ** 2)
    rmse = np.sqrt(mse)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Visualize results
    print("Generating visualizations...")
    visualize_results(model, X_test, y_test, mean, std)