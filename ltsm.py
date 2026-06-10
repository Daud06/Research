import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# 1. Data Preparation Functions (for daily data)
# ==========================================
STEPS_AHEAD = 5       # Forecast horizon (days)
SEQ_LENGTH = 30       # Input sequence length (days)
TWO_WEEKS_DAYS = 14   # Window for rolling median (14 days)

def clean_price_column(series):
    """Convert price strings like '1,457.60' to float."""
    return series.str.replace(',', '').astype(float)

def prepare_and_scale_data(csv_path):
    print(f"Loading and processing {csv_path}...")
    data = pd.read_csv(csv_path, parse_dates=['Date']).set_index('Date').sort_index()

    # Clean numeric columns (remove commas and convert to float)
    data['Price'] = clean_price_column(data['Price'].astype(str))
    data['Open'] = clean_price_column(data['Open'].astype(str))
    data['High'] = clean_price_column(data['High'].astype(str))
    data['Low'] = clean_price_column(data['Low'].astype(str))

    # Clean 'Change %' column: remove '%' and convert to float
    data['Change %'] = data['Change %'].astype(str).str.replace('%', '').astype(float) / 100.0

    # Feature Engineering
    data['day_of_week'] = data.index.dayofweek / 6.0   # Monday=0..Sunday=6, scaled to [0,1]
    data['month'] = data.index.month / 12.0            # month 1..12 scaled to [0,1]

    # Use 'Price' as target. Alternative: 'Change %' – can be changed below.
    target_col = 'Price'   # or 'Change %'

    # Paper-specified step 1: Moving median scaling shifted forward to eliminate future leakage
    rolling_median = data[target_col].rolling(window=TWO_WEEKS_DAYS, min_periods=1).median()
    shifted_median = rolling_median.shift(STEPS_AHEAD)

    data['target_scaled'] = data[target_col] / shifted_median.replace(0, np.nan)
    data = data.dropna(subset=['target_scaled'])

    # Chronological 80-20 train/test split
    split_index = int(len(data) * 0.8)
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    # Paper-specified step 2: Normalize between [0,1] using ONLY training set max bound
    train_max = train_data['target_scaled'].max()
    train_data['target_final'] = train_data['target_scaled'] / train_max
    test_data['target_final'] = test_data['target_scaled'] / train_max

    return train_data, test_data, train_max

def create_sequences(data, feature_cols, target_col):
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH - STEPS_AHEAD + 1):
        X.append(data[feature_cols].iloc[i:i+SEQ_LENGTH].values)
        y.append(data[target_col].iloc[i+SEQ_LENGTH:i+SEQ_LENGTH+STEPS_AHEAD].values)
    return np.array(X), np.array(y)

# ==========================================
# 2. Build LSTM Model
# ==========================================
def build_lstm_model(input_shape):
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
# 3. Model Saving Functions (unchanged, but model_name will be 'lstm_sp500')
# ==========================================
def save_model_in_multiple_formats(model, model_name, save_dir='saved_models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    keras_path = os.path.join(save_dir, f"{model_name}_{timestamp}.keras")
    model.save(keras_path)
    print(f"✅ Saved Keras format: {keras_path}")

    root_keras_path = f"best_{model_name}.keras"
    model.save(root_keras_path)
    print(f"✅ Saved as root model: {root_keras_path}")

    h5_path = os.path.join(save_dir, f"{model_name}_{timestamp}.h5")
    model.save(h5_path)
    print(f"✅ Saved H5 format: {h5_path}")

    json_path = os.path.join(save_dir, f"{model_name}_{timestamp}_architecture.json")
    with open(json_path, 'w') as json_file:
        json_file.write(model.to_json())
    print(f"✅ Saved architecture JSON: {json_path}")

    weights_path = os.path.join(save_dir, f"{model_name}_{timestamp}.weights.h5")
    model.save_weights(weights_path)
    print(f"✅ Saved weights only: {weights_path}")

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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(save_dir, f"training_history_{timestamp}.csv")
    history_df.to_csv(history_path, index=False)
    print(f"✅ Saved training history: {history_path}")

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

    summary_path = os.path.join(save_dir, f"model_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
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
    y_pred = model.predict(X_test, verbose=0)
    y_test_actual = y_test * train_max
    y_pred_actual = y_pred * train_max

    mae = mean_absolute_error(y_test_actual.flatten(), y_pred_actual.flatten())
    r2 = r2_score(y_test_actual.flatten(), y_pred_actual.flatten())

    per_step_mae = []
    for i in range(STEPS_AHEAD):
        step_mae = mean_absolute_error(y_test_actual[:, i], y_pred_actual[:, i])
        per_step_mae.append(step_mae)

    print(f"\n{'='*50}")
    print(f"{model_name} Model Results (daily data):")
    print(f"{'='*50}")
    print(f"Overall MAE: {mae:.4f}")
    print(f"Overall R² Score: {r2:.4f}")
    print(f"\nPer-day MAE (days 1-{STEPS_AHEAD}):")
    for i, step_mae in enumerate(per_step_mae, 1):
        print(f"  Day {i:2d}: {step_mae:.4f}")

    return {
        'mae': mae,
        'r2': r2,
        'per_step_mae': per_step_mae,
        'y_test': y_test_actual,
        'y_pred': y_pred_actual
    }

# ==========================================
# 5. Plotting Functions (updated labels)
# ==========================================
def plot_predictions(results, save_dir='saved_models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    samples_to_plot = min(200, len(results['y_test']))
    actual_flat = results['y_test'].flatten()[:samples_to_plot * STEPS_AHEAD]
    pred_flat = results['y_pred'].flatten()[:samples_to_plot * STEPS_AHEAD]

    axes[0,0].plot(actual_flat, label='Actual', alpha=0.7, linewidth=1)
    axes[0,0].plot(pred_flat, label='LSTM Prediction', alpha=0.7, linewidth=1)
    axes[0,0].set_title(f'LSTM: Actual vs Predicted (MAE: {results["mae"]:.4f})')
    axes[0,0].set_xlabel('Time Steps (Days)')
    axes[0,0].set_ylabel('S&P 500 Price')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].scatter(actual_flat, pred_flat, alpha=0.3, s=1)
    axes[0,1].plot([actual_flat.min(), actual_flat.max()],
                   [actual_flat.min(), actual_flat.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0,1].set_title(f'Scatter Plot (R²: {results["r2"]:.4f})')
    axes[0,1].set_xlabel('Actual Price')
    axes[0,1].set_ylabel('Predicted Price')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    hours = range(1, STEPS_AHEAD+1)
    axes[1,0].bar(hours, results['per_step_mae'], alpha=0.7, color='steelblue')
    axes[1,0].set_title(f'Per-Day MAE ({STEPS_AHEAD}-day forecast)')
    axes[1,0].set_xlabel('Forecast Day')
    axes[1,0].set_ylabel('Mean Absolute Error')
    axes[1,0].grid(True, alpha=0.3, axis='y')

    errors = results['y_pred'].flatten() - results['y_test'].flatten()
    axes[1,1].hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1,1].set_title(f'Prediction Error Distribution (Std: {errors.std():.4f})')
    axes[1,1].set_xlabel('Prediction Error')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'lstm_sp500_predictions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Plot saved as: {plot_path}")
    plt.savefig('lstm_sp500_predictions.png', dpi=300, bbox_inches='tight')

def plot_training_history(history, save_dir='saved_models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'lstm_sp500_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Training history plot saved as: {plot_path}")
    plt.savefig('lstm_sp500_training_history.png', dpi=300, bbox_inches='tight')

# ==========================================
# 6. Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    csv_file_path = 'SP500_merged.csv'
    SAVE_DIR = 'saved_models_sp500'

    if not os.path.exists(csv_file_path):
        print(f"❌ Error: Data file '{csv_file_path}' not found!")
        exit(1)

    print("="*60)
    print("DATA PREPARATION (S&P 500 Daily Data)")
    print("="*60)
    train_df, test_df, train_max = prepare_and_scale_data(csv_file_path)

    # Features: day_of_week, month, and target_final (scaled price)
    features = ['target_final', 'day_of_week', 'month']
    print("\nCreating sequences...")
    X_train, y_train = create_sequences(train_df, features, 'target_final')
    X_test, y_test = create_sequences(test_df, features, 'target_final')

    print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Testing data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Input features: {features}")

    print("\n" + "="*60)
    print("BUILDING LSTM MODEL")
    print("="*60)
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6),
        ModelCheckpoint('best_lstm_sp500.keras', monitor='val_loss', save_best_only=True, verbose=1, mode='min'),
        ModelCheckpoint(os.path.join(SAVE_DIR, 'best_lstm_sp500_checkpoint.keras'), monitor='val_loss', save_best_only=True, verbose=0, mode='min')
    ]

    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    history = model.fit(
        X_train, y_train,
        validation_split=0.20,
        epochs=50,
        batch_size=32,   # smaller batch size for daily data
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*60)
    print("SAVING MODEL IN MULTIPLE FORMATS")
    print("="*60)
    saved_paths = save_model_in_multiple_formats(model, 'lstm_sp500', save_dir=SAVE_DIR)

    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    results = evaluate_model(model, X_test, y_test, train_max, "LSTM")

    checkpoint_paths = save_model_checkpoints(model, history, results, X_test, y_test, save_dir=SAVE_DIR)

    # Save results to CSV
    results_df = pd.DataFrame({'Metric': ['MAE', 'R2'], 'Value': [results['mae'], results['r2']]})
    results_df.to_csv(os.path.join(SAVE_DIR, 'lstm_sp500_results.csv'), index=False)
    results_df.to_csv('lstm_sp500_results.csv', index=False)

    per_hour_df = pd.DataFrame({'Day': range(1, STEPS_AHEAD+1), 'MAE': results['per_step_mae']})
    per_hour_df.to_csv(os.path.join(SAVE_DIR, 'lstm_sp500_per_day_mae.csv'), index=False)

    predictions_df = pd.DataFrame({
        'Actual': results['y_test'].flatten(),
        'Predicted': results['y_pred'].flatten(),
        'Error': results['y_pred'].flatten() - results['y_test'].flatten()
    })
    predictions_df.to_csv(os.path.join(SAVE_DIR, 'lstm_sp500_predictions.csv'), index=False)

    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    plot_training_history(history, save_dir=SAVE_DIR)
    plot_predictions(results, save_dir=SAVE_DIR)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Model Type: LSTM (daily S&P 500)")
    print(f"Sequence Length: {SEQ_LENGTH} days")
    print(f"Forecast Horizon: {STEPS_AHEAD} days")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"\nModel Performance:")
    print(f"  • MAE: {results['mae']:.4f}")
    print(f"  • R² Score: {results['r2']:.4f}")
    print(f"  • Best MAE among {STEPS_AHEAD} days: {min(results['per_step_mae']):.4f}")
    print(f"  • Worst MAE among {STEPS_AHEAD} days: {max(results['per_step_mae']):.4f}")

    print("\n" + "="*60)
    print("✅ LSTM MODEL FOR S&P 500 COMPLETED SUCCESSFULLY!")
    print("="*60)

    # Display saved files
    print("\n📁 All saved files:")
    print("-" * 60)
    print("\nRoot directory:")
    root_files = ['best_lstm_sp500.keras', 'lstm_sp500_results.csv', 'lstm_sp500_predictions.png', 'lstm_sp500_training_history.png']
    for file in root_files:
        if os.path.exists(file):
            size = os.path.getsize(file)/1024
            print(f"  • {file:<35} ({size:.2f} KB)")

    if os.path.exists(SAVE_DIR):
        print(f"\n{SAVE_DIR} directory:")
        for file in os.listdir(SAVE_DIR):
            fpath = os.path.join(SAVE_DIR, file)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)/1024
                print(f"  • {file:<45} ({size:.2f} KB)")