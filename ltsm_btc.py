import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# 1. Data Preparation Functions
# ==========================================
STEPS_AHEAD = 24  # Number of hours to forecast ahead
SEQ_LENGTH = 168  # Input sequence window (1 week / 168 hours)
TWO_WEEKS_HOURS = 336  # Context window for rolling median scaling


def prepare_and_scale_data(csv_path):
    print(f"Loading and processing {csv_path}...")
    # Load and explicitly handle Timestamp parsing
    data = pd.read_csv(csv_path, parse_dates=['Timestamp']).set_index('Timestamp').sort_index()

    # Feature Engineering: Extract temporal sequences known in the future
    data['hour_of_day'] = data.index.hour / 23.0
    data['day_of_week'] = data.index.dayofweek / 6.0

    # Paper-specified step 1: Moving median scaling shifted forward to eliminate future leakage
    rolling_median = data['Quote_Asset_Volume'].rolling(window=TWO_WEEKS_HOURS, min_periods=1).median()
    shifted_median = rolling_median.shift(STEPS_AHEAD)

    data['target_scaled'] = data['Quote_Asset_Volume'] / shifted_median.replace(0, np.nan)
    data = data.dropna(subset=['target_scaled'])

    # Chronological 80-20 train/test split
    split_index = int(len(data) * 0.8)
    train_data, test_data = data.iloc[:split_index], data.iloc[split_index:]

    # Paper-specified step 2: Normalize between [0, 1] using ONLY training set max bound
    train_max = train_data['target_scaled'].max()
    train_data = train_data.copy()
    test_data = test_data.copy()
    train_data['target_final'] = train_data['target_scaled'] / train_max
    test_data['target_final'] = test_data['target_scaled'] / train_max

    return train_data, test_data, train_max


def create_sequences(data, feature_cols, target_col):
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH - STEPS_AHEAD + 1):
        X.append(data[feature_cols].iloc[i: i + SEQ_LENGTH].values)
        y.append(data[target_col].iloc[i + SEQ_LENGTH: i + SEQ_LENGTH + STEPS_AHEAD].values)
    return np.array(X), np.array(y)


# ==========================================
# 2. Build LSTM Model
# ==========================================
def build_lstm_model(input_shape):
    """Build standard LSTM model for time series forecasting"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(100, return_sequences=True),
        LSTM(100, return_sequences=False),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(STEPS_AHEAD, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ==========================================
# 3. Model Saving Functions (FIXED)
# ==========================================
def save_model_in_multiple_formats(model, model_name, save_dir='saved_models'):
    """
    Save model in multiple formats for different use cases
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save in native Keras format (recommended) - THIS IS THE MAIN MODEL FILE
    keras_path = os.path.join(save_dir, f"{model_name}_{timestamp}.keras")
    model.save(keras_path)
    print(f"✅ Saved Keras format: {keras_path}")

    # Also save as 'best_lstm_model.keras' in root directory for easy access
    root_keras_path = f"best_{model_name}.keras"
    model.save(root_keras_path)
    print(f"✅ Saved as root model: {root_keras_path}")

    # 2. Save in H5 format (legacy/compatibility)
    h5_path = os.path.join(save_dir, f"{model_name}_{timestamp}.h5")
    model.save(h5_path)
    print(f"✅ Saved H5 format: {h5_path}")

    # 3. Save model architecture as JSON
    json_path = os.path.join(save_dir, f"{model_name}_{timestamp}_architecture.json")
    with open(json_path, 'w') as json_file:
        json_file.write(model.to_json())
    print(f"✅ Saved architecture JSON: {json_path}")

    # 4. Save weights only - FIXED: Use .weights.h5 extension
    weights_path = os.path.join(save_dir, f"{model_name}_{timestamp}.weights.h5")
    model.save_weights(weights_path)
    print(f"✅ Saved weights only: {weights_path}")

    # 5. Save as TensorFlow SavedModel format
    saved_model_path = os.path.join(save_dir, f"{model_name}_{timestamp}_saved_model")
    tf.saved_model.save(model, saved_model_path)
    print(f"✅ Saved TensorFlow SavedModel: {saved_model_path}")

    return {
        'keras_path': keras_path,
        'root_keras_path': root_keras_path,
        'h5_path': h5_path,
        'json_path': json_path,
        'weights_path': weights_path,
        'saved_model_path': saved_model_path,
        'timestamp': timestamp
    }


def save_model_checkpoints(model, history, results, X_test, y_test, save_dir='saved_models'):
    """
    Save additional model artifacts and metadata
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save training history
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(save_dir, f"training_history_{timestamp}.csv")
    history_df.to_csv(history_path, index=False)
    print(f"✅ Saved training history: {history_path}")

    # 2. Save model metadata and parameters
    metadata = {
        'model_type': 'LSTM',
        'sequence_length': SEQ_LENGTH,
        'steps_ahead': STEPS_AHEAD,
        'timestamp': timestamp,
        'training_samples': len(X_test) if X_test is not None else None,
        'lstm_units': 100,
        'dense_layers': [50, 25],
        'optimizer': 'adam',
        'loss_function': 'mse',
        'metrics': ['mae'],
        'final_mae': float(results['mae']) if results else None,
        'final_r2': float(results['r2']) if results else None
    }

    metadata_path = os.path.join(save_dir, f"model_metadata_{timestamp}.json")
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Saved model metadata: {metadata_path}")

    # 3. Save model summary as text
    summary_path = os.path.join(save_dir, f"model_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        # Capture model summary
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"✅ Saved model summary: {summary_path}")

    return {
        'history_path': history_path,
        'metadata_path': metadata_path,
        'summary_path': summary_path
    }


# ==========================================
# 4. Evaluation Metrics Function
# ==========================================
def evaluate_model(model, X_test, y_test, train_max, model_name="LSTM"):
    """
    Evaluate model and return MAE and R² metrics
    """
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)

    # Inverse scaling to get actual values
    y_test_actual = y_test * train_max
    y_pred_actual = y_pred * train_max

    # Calculate metrics
    mae = mean_absolute_error(y_test_actual.flatten(), y_pred_actual.flatten())
    r2 = r2_score(y_test_actual.flatten(), y_pred_actual.flatten())

    # Calculate per-step MAE (for each of the 24 forecast hours)
    per_step_mae = []
    for i in range(STEPS_AHEAD):
        step_mae = mean_absolute_error(y_test_actual[:, i], y_pred_actual[:, i])
        per_step_mae.append(step_mae)

    print(f"\n{'=' * 50}")
    print(f"{model_name} Model Results:")
    print(f"{'=' * 50}")
    print(f"Overall MAE: {mae:.4f}")
    print(f"Overall R² Score: {r2:.4f}")
    print(f"\nPer-hour MAE (hours 1-24):")
    for i, step_mae in enumerate(per_step_mae, 1):
        print(f"  Hour {i:2d}: {step_mae:.4f}")

    return {
        'mae': mae,
        'r2': r2,
        'per_step_mae': per_step_mae,
        'y_test': y_test_actual,
        'y_pred': y_pred_actual
    }


# ==========================================
# 5. Plotting Functions
# ==========================================
def plot_predictions(results, save_dir='saved_models'):
    """
    Plot LSTM predictions vs actual values
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Predictions vs Actual (first 200 samples)
    ax1 = axes[0, 0]
    samples_to_plot = min(200, len(results['y_test']))

    # Flatten for plotting
    actual_flat = results['y_test'].flatten()[:samples_to_plot * STEPS_AHEAD]
    pred_flat = results['y_pred'].flatten()[:samples_to_plot * STEPS_AHEAD]

    ax1.plot(actual_flat, label='Actual', alpha=0.7, linewidth=1)
    ax1.plot(pred_flat, label='LSTM Prediction', alpha=0.7, linewidth=1)
    ax1.set_title(f'LSTM Model: Actual vs Predicted (MAE: {results["mae"]:.4f})')
    ax1.set_xlabel('Time Steps (Hours)')
    ax1.set_ylabel('Quote Asset Volume')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot of predictions vs actual
    ax2 = axes[0, 1]
    ax2.scatter(actual_flat, pred_flat, alpha=0.3, s=1)
    ax2.plot([actual_flat.min(), actual_flat.max()],
             [actual_flat.min(), actual_flat.max()],
             'r--', lw=2, label='Perfect Prediction')
    ax2.set_title(f'Scatter Plot: Actual vs Predicted (R²: {results["r2"]:.4f})')
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Per-hour MAE
    ax3 = axes[1, 0]
    hours = range(1, STEPS_AHEAD + 1)
    ax3.bar(hours, results['per_step_mae'], alpha=0.7, color='steelblue')
    ax3.set_title('Per-Hour MAE (24-hour forecast)')
    ax3.set_xlabel('Forecast Hour')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Error distribution
    ax4 = axes[1, 1]
    errors = results['y_pred'].flatten() - results['y_test'].flatten()
    ax4.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.set_title(f'Prediction Error Distribution (Std: {errors.std():.4f})')
    ax4.set_xlabel('Prediction Error')
    ax4.set_ylabel('Frequency')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, 'lstm_predictions_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Plot saved as: {plot_path}")

    # Also save in root directory
    plt.savefig('lstm_predictions_plot.png', dpi=300, bbox_inches='tight')


def plot_training_history(history, save_dir='saved_models'):
    """
    Plot training history
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot MAE
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, 'lstm_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Training history plot saved as: {plot_path}")

    # Also save in root directory
    plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')


# ==========================================
# 6. Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    csv_file_path = 'btc.csv'
    SAVE_DIR = 'saved_models'

    # Check if data file exists
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: Data file '{csv_file_path}' not found!")
        exit(1)

    # Prepare data
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    train_df, test_df, train_max = prepare_and_scale_data(csv_file_path)

    features = ['target_final', 'hour_of_day', 'day_of_week']

    # Create sequences
    print("\nCreating sequences...")
    X_train, y_train = create_sequences(train_df, features, 'target_final')
    X_test, y_test = create_sequences(test_df, features, 'target_final')

    print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Testing data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Input features: {features}")

    # Build LSTM model
    print("\n" + "=" * 60)
    print("BUILDING LSTM MODEL")
    print("=" * 60)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    # Create callbacks for training with model checkpointing
    # FIXED: Save checkpoint with correct extension
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6),
        # Save best model during training - this creates 'best_lstm_model.keras'
        ModelCheckpoint('best_lstm_model.keras',
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1,
                        mode='min'),
        # Also save checkpoint in saved_models directory
        ModelCheckpoint(os.path.join(SAVE_DIR, 'best_lstm_checkpoint.keras'),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=0,
                        mode='min')
    ]

    # Train the model
    print("\n" + "=" * 60)
    print("TRAINING LSTM MODEL")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        validation_split=0.20,
        epochs=15,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # ==========================================
    # SAVE MODEL IN MULTIPLE FORMATS
    # ==========================================
    print("\n" + "=" * 60)
    print("SAVING MODEL IN MULTIPLE FORMATS")
    print("=" * 60)

    # Save final model in multiple formats (this will also save as 'best_lstm_model.keras')
    saved_paths = save_model_in_multiple_formats(model, 'lstm_model', save_dir=SAVE_DIR)

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    results = evaluate_model(model, X_test, y_test, train_max, "LSTM")

    # Save model checkpoints and metadata with results
    checkpoint_paths = save_model_checkpoints(model, history, results, X_test, y_test, save_dir=SAVE_DIR)

    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['MAE', 'R2'],
        'Value': [results['mae'], results['r2']]
    })
    results_csv_path = os.path.join(SAVE_DIR, 'lstm_model_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"✅ Results saved to: {results_csv_path}")

    # Also save in root directory
    results_df.to_csv('lstm_model_results.csv', index=False)

    # Save per-hour MAE to CSV
    per_hour_df = pd.DataFrame({
        'Hour': range(1, STEPS_AHEAD + 1),
        'MAE': results['per_step_mae']
    })
    per_hour_csv_path = os.path.join(SAVE_DIR, 'lstm_per_hour_mae.csv')
    per_hour_df.to_csv(per_hour_csv_path, index=False)
    print(f"✅ Per-hour MAE saved to: {per_hour_csv_path}")

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Actual': results['y_test'].flatten(),
        'Predicted': results['y_pred'].flatten(),
        'Error': results['y_pred'].flatten() - results['y_test'].flatten()
    })
    predictions_csv_path = os.path.join(SAVE_DIR, 'lstm_predictions.csv')
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"✅ Predictions saved to: {predictions_csv_path}")

    # Plot results
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_training_history(history, save_dir=SAVE_DIR)
    plot_predictions(results, save_dir=SAVE_DIR)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Model Type: LSTM (Long Short-Term Memory)")
    print(f"Sequence Length: {SEQ_LENGTH} hours")
    print(f"Forecast Horizon: {STEPS_AHEAD} hours")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"\nModel Performance:")
    print(f"  • MAE: {results['mae']:.4f}")
    print(f"  • R² Score: {results['r2']:.4f}")
    print(f"  • Best MAE among 24 hours: {min(results['per_step_mae']):.4f}")
    print(f"  • Worst MAE among 24 hours: {max(results['per_step_mae']):.4f}")

    print("\n" + "=" * 60)
    print("✅ LSTM MODEL TRAINING, EVALUATION, AND SAVING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    # Display all saved files
    print("\n📁 All saved files:")
    print("-" * 60)

    # Root directory files
    print("\nRoot directory:")
    root_files = ['best_lstm_model.keras', 'lstm_model_results.csv',
                  'lstm_predictions_plot.png', 'lstm_training_history.png']
    for file in root_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / 1024
            print(f"  • {file:<35} ({file_size:.2f} KB)")

    # Saved models directory
    if os.path.exists(SAVE_DIR):
        print(f"\n{SAVE_DIR} directory:")
        for file in os.listdir(SAVE_DIR):
            file_path = os.path.join(SAVE_DIR, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path) / 1024
                print(f"  • {file:<45} ({file_size:.2f} KB)")

    print("\n" + "=" * 60)
    print("IMPORTANT: Model saved as 'best_lstm_model.keras'")
    print("You can now use this file for evaluation with TKAN model")
    print("=" * 60)