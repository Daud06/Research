import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# 1. Константы (для дневных данных)
# ==========================================
STEPS_AHEAD = 5       # Горизонт прогноза (дней)
SEQ_LENGTH = 30       # Длина входной последовательности (дней)
TWO_WEEKS_DAYS = 14   # Окно скользящей медианы (14 дней)

# ==========================================
# 2. KAN Basis Layer
# ==========================================
class KANBasisLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, num_basis=5, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis

    def build(self, input_shape):
        self.w_base = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="w_base"
        )
        self.w_spline = self.add_weight(
            shape=(self.num_basis, self.input_dim, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="w_spline"
        )
        grid_values = np.linspace(-2.0, 2.0, self.num_basis).astype(np.float32)
        self.centers = self.add_weight(
            shape=(self.num_basis,),
            initializer=tf.keras.initializers.Constant(grid_values),
            trainable=False,
            name="centers"
        )
        super().build(input_shape)

    def call(self, x):
        base_output = tf.matmul(x, self.w_base)
        x_expanded = tf.expand_dims(x, axis=-1)
        basis_outputs = tf.exp(-tf.square(x_expanded - self.centers))
        basis_transposed = tf.transpose(basis_outputs, perm=[2, 0, 1])
        spline_output = tf.zeros_like(base_output)
        for i in range(self.num_basis):
            spline_output += tf.matmul(basis_transposed[i], self.w_spline[i])
        return base_output + spline_output

# ==========================================
# 3. TKAN Recurrent Layer (custom)
# ==========================================
class CustomTKAN(tf.keras.layers.Layer):
    def __init__(self, units, return_sequences=False, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.num_layers = num_layers

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # LSTM‑подобные гейты
        self.W_f = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="W_f")
        self.U_f = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="U_f")
        self.b_f = self.add_weight(shape=(self.units,), initializer="zeros", name="b_f")

        self.W_i = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="W_i")
        self.U_i = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="U_i")
        self.b_i = self.add_weight(shape=(self.units,), initializer="zeros", name="b_i")

        self.W_c = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="W_c")
        self.U_c = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="U_c")
        self.b_c = self.add_weight(shape=(self.units,), initializer="zeros", name="b_c")

        # Веса для RKAN (KAN‑адаптация)
        self.W_x_tilde = [
            self.add_weight(shape=(input_dim, input_dim), initializer="glorot_uniform", name=f"W_x_tilde_{i}")
            for i in range(self.num_layers)
        ]
        self.W_h_tilde = [
            self.add_weight(shape=(self.units, input_dim), initializer="glorot_uniform", name=f"W_h_tilde_{i}")
            for i in range(self.num_layers)
        ]

        self.W_hh = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="W_hh")
        self.W_hz = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="W_hz")

        self.kan_layers = [KANBasisLayer(input_dim=input_dim, output_dim=self.units) for _ in range(self.num_layers)]

        self.W_o = self.add_weight(shape=(self.units * self.num_layers, self.units), initializer="glorot_uniform", name="W_o")
        self.b_o = self.add_weight(shape=(self.units,), initializer="zeros", name="b_o")

        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = inputs.shape[1]

        h_t = tf.zeros((batch_size, self.units))
        c_t = tf.zeros((batch_size, self.units))
        h_tilde_layers = [tf.zeros((batch_size, self.units)) for _ in range(self.num_layers)]

        output_sequences = tf.TensorArray(dtype=tf.float32, size=time_steps)

        for t in range(time_steps):
            x_t = inputs[:, t, :]

            f_t = tf.sigmoid(tf.matmul(x_t, self.W_f) + tf.matmul(h_t, self.U_f) + self.b_f)
            i_t = tf.sigmoid(tf.matmul(x_t, self.W_i) + tf.matmul(h_t, self.U_i) + self.b_i)
            c_tilde_t = tf.sigmoid(tf.matmul(x_t, self.W_c) + tf.matmul(h_t, self.U_c) + self.b_c)

            kan_outputs = []
            for l in range(self.num_layers):
                s_lt = tf.matmul(x_t, self.W_x_tilde[l]) + tf.matmul(h_tilde_layers[l], self.W_h_tilde[l])
                o_tilde_t = self.kan_layers[l](s_lt)
                kan_outputs.append(o_tilde_t)
                h_tilde_layers[l] = tf.matmul(h_tilde_layers[l], self.W_hh) + tf.matmul(o_tilde_t, self.W_hz)

            r_t = tf.concat(kan_outputs, axis=-1)
            o_t = tf.sigmoid(tf.matmul(r_t, self.W_o) + self.b_o)

            c_t = f_t * c_t + i_t * c_tilde_t
            h_t = o_t * tf.tanh(c_t)

            output_sequences = output_sequences.write(t, h_t)

        output_sequences = tf.transpose(output_sequences.stack(), perm=[1, 0, 2])
        if self.return_sequences:
            return output_sequences
        else:
            return output_sequences[:, -1, :]

# ==========================================
# 4. Подготовка данных (для дневных данных S&P 500)
# ==========================================
def clean_price_column(series):
    return series.str.replace(',', '').astype(float)

def prepare_and_scale_data(csv_path):
    print(f"Loading and processing {csv_path}...")
    data = pd.read_csv(csv_path, parse_dates=['Date']).set_index('Date').sort_index()

    data['Price'] = clean_price_column(data['Price'].astype(str))
    data['Open'] = clean_price_column(data['Open'].astype(str))
    data['High'] = clean_price_column(data['High'].astype(str))
    data['Low'] = clean_price_column(data['Low'].astype(str))
    data['Change %'] = data['Change %'].astype(str).str.replace('%', '').astype(float) / 100.0

    # Признаки: день недели, месяц
    data['day_of_week'] = data.index.dayofweek / 6.0
    data['month'] = data.index.month / 12.0

    target_col = 'Price'   # можно заменить на 'Change %'

    rolling_median = data[target_col].rolling(window=TWO_WEEKS_DAYS, min_periods=1).median()
    shifted_median = rolling_median.shift(STEPS_AHEAD)

    data['target_scaled'] = data[target_col] / shifted_median.replace(0, np.nan)
    data = data.dropna(subset=['target_scaled'])

    split_index = int(len(data) * 0.8)
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

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
# 5. Оценка и визуализация
# ==========================================
def evaluate_model(model, X_test, y_test, train_max, model_name="TKAN"):
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

def plot_predictions(results, save_dir='saved_models_tkan'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    samples_to_plot = min(200, len(results['y_test']))
    actual_flat = results['y_test'].flatten()[:samples_to_plot * STEPS_AHEAD]
    pred_flat = results['y_pred'].flatten()[:samples_to_plot * STEPS_AHEAD]

    axes[0,0].plot(actual_flat, label='Actual', alpha=0.7, linewidth=1)
    axes[0,0].plot(pred_flat, label='TKAN Prediction', alpha=0.7, linewidth=1)
    axes[0,0].set_title(f'TKAN: Actual vs Predicted (MAE: {results["mae"]:.4f})')
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

    days = range(1, STEPS_AHEAD+1)
    axes[1,0].bar(days, results['per_step_mae'], alpha=0.7, color='purple')
    axes[1,0].set_title(f'Per-Day MAE ({STEPS_AHEAD}-day forecast)')
    axes[1,0].set_xlabel('Forecast Day')
    axes[1,0].set_ylabel('Mean Absolute Error')
    axes[1,0].grid(True, alpha=0.3, axis='y')

    errors = results['y_pred'].flatten() - results['y_test'].flatten()
    axes[1,1].hist(errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].set_title(f'Prediction Error Distribution (Std: {errors.std():.4f})')
    axes[1,1].set_xlabel('Prediction Error')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'tkan_sp500_predictions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Plot saved as: {plot_path}")
    plt.savefig('tkan_sp500_predictions.png', dpi=300, bbox_inches='tight')

def plot_training_history(history, save_dir='saved_models_tkan'):
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
    plot_path = os.path.join(save_dir, 'tkan_sp500_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Training history plot saved as: {plot_path}")
    plt.savefig('tkan_sp500_training_history.png', dpi=300, bbox_inches='tight')

# ==========================================
# 6. Сохранение модели и метаданных
# ==========================================
def save_model_in_multiple_formats(model, model_name, save_dir='saved_models_tkan'):
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

def save_model_checkpoints(model, history, results, X_test, y_test, save_dir='saved_models_tkan'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(save_dir, f"training_history_{timestamp}.csv")
    history_df.to_csv(history_path, index=False)
    print(f"✅ Saved training history: {history_path}")

    metadata = {
        'model_type': 'TKAN',
        'sequence_length': SEQ_LENGTH,
        'steps_ahead': STEPS_AHEAD,
        'timestamp': timestamp,
        'training_samples': len(X_test) if X_test is not None else None,
        'tkan_units': 100,
        'num_layers': 2,
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
# 7. Основной блок
# ==========================================
if __name__ == "__main__":
    csv_file_path = 'SP500_merged.csv'
    SAVE_DIR = 'saved_models_tkan'

    if not os.path.exists(csv_file_path):
        print(f"❌ Error: Data file '{csv_file_path}' not found!")
        exit(1)

    print("="*60)
    print("DATA PREPARATION (S&P 500 Daily Data)")
    print("="*60)
    train_df, test_df, train_max = prepare_and_scale_data(csv_file_path)

    features = ['target_final', 'day_of_week', 'month']
    print("\nCreating sequences...")
    X_train, y_train = create_sequences(train_df, features, 'target_final')
    X_test, y_test = create_sequences(test_df, features, 'target_final')

    print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Testing data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Input features: {features}")

    print("\n" + "="*60)
    print("BUILDING TKAN MODEL")
    print("="*60)
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        CustomTKAN(units=100, return_sequences=True, num_layers=2),
        CustomTKAN(units=100, return_sequences=False, num_layers=2),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(STEPS_AHEAD, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6),
        ModelCheckpoint('best_tkan_sp500.keras', monitor='val_loss', save_best_only=True, verbose=1, mode='min'),
        ModelCheckpoint(os.path.join(SAVE_DIR, 'best_tkan_sp500_checkpoint.keras'), monitor='val_loss', save_best_only=True, verbose=0, mode='min')
    ]

    print("\n" + "="*60)
    print("TRAINING TKAN MODEL")
    print("="*60)
    history = model.fit(
        X_train, y_train,
        validation_split=0.20,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*60)
    print("SAVING MODEL IN MULTIPLE FORMATS")
    print("="*60)
    saved_paths = save_model_in_multiple_formats(model, 'tkan_sp500', save_dir=SAVE_DIR)

    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    results = evaluate_model(model, X_test, y_test, train_max, "TKAN")

    save_model_checkpoints(model, history, results, X_test, y_test, save_dir=SAVE_DIR)

    # Сохраняем результаты в CSV
    results_df = pd.DataFrame({'Metric': ['MAE', 'R2'], 'Value': [results['mae'], results['r2']]})
    results_df.to_csv(os.path.join(SAVE_DIR, 'tkan_sp500_results.csv'), index=False)
    results_df.to_csv('tkan_sp500_results.csv', index=False)

    per_day_df = pd.DataFrame({'Day': range(1, STEPS_AHEAD+1), 'MAE': results['per_step_mae']})
    per_day_df.to_csv(os.path.join(SAVE_DIR, 'tkan_sp500_per_day_mae.csv'), index=False)

    predictions_df = pd.DataFrame({
        'Actual': results['y_test'].flatten(),
        'Predicted': results['y_pred'].flatten(),
        'Error': results['y_pred'].flatten() - results['y_test'].flatten()
    })
    predictions_df.to_csv(os.path.join(SAVE_DIR, 'tkan_sp500_predictions.csv'), index=False)

    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    plot_training_history(history, save_dir=SAVE_DIR)
    plot_predictions(results, save_dir=SAVE_DIR)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Model Type: TKAN (Kolmogorov–Arnold Network)")
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
    print("✅ TKAN MODEL FOR S&P 500 COMPLETED SUCCESSFULLY!")
    print("="*60)

    print("\n📁 All saved files:")
    print("-" * 60)
    print("\nRoot directory:")
    root_files = ['best_tkan_sp500.keras', 'tkan_sp500_results.csv', 'tkan_sp500_predictions.png', 'tkan_sp500_training_history.png']
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