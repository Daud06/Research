import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# КОНСТАНТЫ (дневные данные S&P 500)
# =====================================================

STEPS_AHEAD = 5        # прогноз на 5 дней вперёд
SEQ_LENGTH = 30        # входная последовательность (30 дней)
TWO_WEEKS_DAYS = 14    # окно скользящей медианы (14 дней)


# =====================================================
# KAN Layer (для TKAN)
# =====================================================
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


# =====================================================
# TKAN Recurrent Layer
# =====================================================
class CustomTKAN(tf.keras.layers.Layer):
    def __init__(self, units, return_sequences=False, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.num_layers = num_layers

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.W_f = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="W_f")
        self.U_f = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="U_f")
        self.b_f = self.add_weight(shape=(self.units,), initializer="zeros", name="b_f")

        self.W_i = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="W_i")
        self.U_i = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="U_i")
        self.b_i = self.add_weight(shape=(self.units,), initializer="zeros", name="b_i")

        self.W_c = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="W_c")
        self.U_c = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="U_c")
        self.b_c = self.add_weight(shape=(self.units,), initializer="zeros", name="b_c")

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
        return output_sequences[:, -1, :]


# =====================================================
# Подготовка данных (дневные S&P 500)
# =====================================================
def clean_price_column(series):
    return series.str.replace(',', '').astype(float)

def prepare_and_scale_data(csv_path):
    data = pd.read_csv(csv_path, parse_dates=['Date']).set_index('Date').sort_index()

    # Очистка числовых колонок
    data['Price'] = clean_price_column(data['Price'].astype(str))
    data['Open'] = clean_price_column(data['Open'].astype(str))
    data['High'] = clean_price_column(data['High'].astype(str))
    data['Low'] = clean_price_column(data['Low'].astype(str))
    data['Change %'] = data['Change %'].astype(str).str.replace('%', '').astype(float) / 100.0

    # Признаки: день недели, месяц
    data['day_of_week'] = data.index.dayofweek / 6.0
    data['month'] = data.index.month / 12.0

    target_col = 'Price'   # или 'Change %'

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


# =====================================================
# Функция оценки модели
# =====================================================
def evaluate_model(model, X_test, y_test, train_max, model_name):
    y_pred = model.predict(X_test, verbose=0)
    y_test_actual = y_test * train_max
    y_pred_actual = y_pred * train_max

    mae = mean_absolute_error(y_test_actual.flatten(), y_pred_actual.flatten())
    rmse = np.sqrt(mean_squared_error(y_test_actual.flatten(), y_pred_actual.flatten()))
    r2 = r2_score(y_test_actual.flatten(), y_pred_actual.flatten())
    mape = np.mean(np.abs((y_test_actual.flatten() - y_pred_actual.flatten()) / (y_test_actual.flatten() + 1e-8))) * 100

    print(f"\n{'='*60}")
    print(f"{model_name} - TEST METRICS (daily)")
    print(f"{'='*60}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")
    print(f"MAPE : {mape:.2f}%")

    per_day_metrics = []
    for d in range(STEPS_AHEAD):
        r2_d = r2_score(y_test_actual[:, d], y_pred_actual[:, d])
        mae_d = mean_absolute_error(y_test_actual[:, d], y_pred_actual[:, d])
        rmse_d = np.sqrt(mean_squared_error(y_test_actual[:, d], y_pred_actual[:, d]))
        per_day_metrics.append({
            'day': d+1,
            'mae': mae_d,
            'rmse': rmse_d,
            'r2': r2_d
        })

    print(f"\n{model_name} - PER DAY METRICS (first {STEPS_AHEAD} days):")
    print("-"*70)
    print(f"{'Day':<6} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-"*70)
    for m in per_day_metrics:
        print(f"{m['day']:02d}    {m['mae']:<12.6f} {m['rmse']:<12.6f} {m['r2']:<12.6f}")

    return {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'per_day_metrics': per_day_metrics,
        'y_test': y_test_actual,
        'y_pred': y_pred_actual
    }


# =====================================================
# Визуализация сравнения моделей
# =====================================================
def plot_model_comparison(tkan_results, lstm_results=None, gru_results=None):
    available_models = []
    if tkan_results: available_models.append(('TKAN', tkan_results, 'blue'))
    if lstm_results: available_models.append(('LSTM', lstm_results, 'orange'))
    if gru_results: available_models.append(('GRU', gru_results, 'green'))

    if len(available_models) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Первый прогноз
        ax1 = axes[0,0]
        idx = 0
        ax1.plot(tkan_results['y_test'][idx], label='Actual', linewidth=2, alpha=0.8)
        for name, res, col in available_models:
            ax1.plot(res['y_pred'][idx], label=name, linewidth=2, alpha=0.8, color=col)
        ax1.set_title(f'{STEPS_AHEAD}-Day Forecast (Sample 1)')
        ax1.set_xlabel('Days Ahead')
        ax1.set_ylabel('S&P 500 Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Второй прогноз
        ax2 = axes[0,1]
        idx = min(5, len(tkan_results['y_test'])-1)
        ax2.plot(tkan_results['y_test'][idx], label='Actual', linewidth=2, alpha=0.8)
        for name, res, col in available_models:
            ax2.plot(res['y_pred'][idx], label=name, linewidth=2, alpha=0.8, color=col)
        ax2.set_title(f'{STEPS_AHEAD}-Day Forecast (Sample 2)')
        ax2.set_xlabel('Days Ahead')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Сравнение основных метрик
        ax3 = axes[0,2]
        metrics = ['MAE', 'RMSE', 'MAPE']
        x = np.arange(len(metrics))
        width = 0.8 / len(available_models)
        for i, (name, res, col) in enumerate(available_models):
            values = [res['mae'], res['rmse'], res['mape']]
            offset = (i - len(available_models)/2 + 0.5) * width
            ax3.bar(x + offset, values, width, label=name, alpha=0.8, color=col)
        ax3.set_title('Model Performance Comparison')
        ax3.set_ylabel('Error Value')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # R² по дням
        ax4 = axes[1,0]
        days = range(1, STEPS_AHEAD+1)
        for name, res, col in available_models:
            r2_day = [m['r2'] for m in res['per_day_metrics']]
            ax4.plot(days, r2_day, marker='o', label=name, linewidth=2, markersize=4, color=col)
        ax4.set_title('R² Score by Forecast Day')
        ax4.set_xlabel('Days Ahead')
        ax4.set_ylabel('R² Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # MAE по дням
        ax5 = axes[1,1]
        for name, res, col in available_models:
            mae_day = [m['mae'] for m in res['per_day_metrics']]
            ax5.plot(days, mae_day, marker='s', label=name, linewidth=2, markersize=4, color=col)
        ax5.set_title('MAE by Forecast Day')
        ax5.set_xlabel('Days Ahead')
        ax5.set_ylabel('Mean Absolute Error')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Распределение ошибок
        ax6 = axes[1,2]
        for name, res, col in available_models:
            errors = (res['y_pred'] - res['y_test']).flatten()
            ax6.hist(errors, bins=50, alpha=0.4, label=f'{name} (std: {errors.std():.4f})',
                     color=col, edgecolor='black')
        ax6.set_title('Prediction Error Distribution')
        ax6.set_xlabel('Prediction Error')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.7)

        plt.tight_layout()
        plt.savefig('sp500_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n✅ Comparison plot saved as 'sp500_model_comparison.png'")

    elif len(available_models) == 1:
        name, res, col = available_models[0]
        fig, axes = plt.subplots(2,2, figsize=(15,10))
        axes[0,0].plot(res['y_test'][0], label='Actual', linewidth=2)
        axes[0,0].plot(res['y_pred'][0], label=f'{name} Prediction', linewidth=2, color=col)
        axes[0,0].set_title(f'{name}: {STEPS_AHEAD}-Day Forecast')
        axes[0,0].set_xlabel('Days Ahead')
        axes[0,0].set_ylabel('Price')
        axes[0,0].legend()
        axes[0,0].grid(True)

        y_true_f = res['y_test'].flatten()
        y_pred_f = res['y_pred'].flatten()
        axes[0,1].scatter(y_true_f, y_pred_f, alpha=0.3, s=1, color=col)
        axes[0,1].plot([y_true_f.min(), y_true_f.max()], [y_true_f.min(), y_true_f.max()], 'r--')
        axes[0,1].set_title(f'Actual vs Predicted (R²={res["r2"]:.4f})')
        axes[0,1].set_xlabel('Actual')
        axes[0,1].set_ylabel('Predicted')
        axes[0,1].grid(True)

        days = range(1, STEPS_AHEAD+1)
        mae_day = [m['mae'] for m in res['per_day_metrics']]
        axes[1,0].bar(days, mae_day, color=col, alpha=0.7)
        axes[1,0].set_title('MAE by Forecast Day')
        axes[1,0].set_xlabel('Days Ahead')
        axes[1,0].set_ylabel('MAE')
        axes[1,0].grid(True, axis='y')

        errors = (res['y_pred'] - res['y_test']).flatten()
        axes[1,1].hist(errors, bins=50, color=col, alpha=0.7, edgecolor='black')
        axes[1,1].set_title(f'Error Distribution (std={errors.std():.4f})')
        axes[1,1].set_xlabel('Error')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].axvline(0, color='r', linestyle='--')
        axes[1,1].grid(True)

        plt.tight_layout()
        plt.savefig(f'sp500_{name.lower()}_evaluation.png', dpi=300)
        plt.show()
        print(f"\n✅ {name} evaluation plot saved as 'sp500_{name.lower()}_evaluation.png'")


# =====================================================
# ОСНОВНАЯ ПРОГРАММА
# =====================================================
if __name__ == "__main__":
    print("="*60)
    print("ЗАГРУЗКА ДАННЫХ S&P 500")
    print("="*60)

    csv_file_path = "SP500_merged.csv"
    if not os.path.exists(csv_file_path):
        print(f"❌ Файл {csv_file_path} не найден!")
        exit(1)

    train_df, test_df, train_max = prepare_and_scale_data(csv_file_path)

    features = ['target_final', 'day_of_week', 'month']
    X_test, y_test = create_sequences(test_df, features, 'target_final')

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Train max scaling factor: {train_max:.6f}")

    # ---------- Поиск моделей ----------
    print("\n" + "="*60)
    print("ПОИСК СОХРАНЁННЫХ МОДЕЛЕЙ")
    print("="*60)

    tkan_paths = ['best_tkan_sp500.keras', 'final_tkan_sp500.keras', 'tkan_sp500_model.keras']
    lstm_paths = ['best_lstm_sp500.keras', 'final_lstm_sp500.keras', 'lstm_sp500_model.keras']
    gru_paths  = ['best_gru_sp500.keras',  'final_gru_sp500.keras',  'gru_sp500_model.keras']

    tkan_file = next((p for p in tkan_paths if os.path.exists(p)), None)
    lstm_file = next((p for p in lstm_paths if os.path.exists(p)), None)
    gru_file  = next((p for p in gru_paths  if os.path.exists(p)), None)

    # ---------- Загрузка TKAN ----------
    tkan_model = None
    if tkan_file:
        print(f"✅ Найден TKAN: {tkan_file}")
        try:
            tkan_model = tf.keras.models.load_model(
                tkan_file,
                custom_objects={'CustomTKAN': CustomTKAN, 'KANBasisLayer': KANBasisLayer}
            )
            print("✅ TKAN загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки TKAN: {e}")
    else:
        print("❌ TKAN модель не найдена")

    # ---------- Загрузка LSTM ----------
    lstm_model = None
    if lstm_file:
        print(f"✅ Найден LSTM: {lstm_file}")
        try:
            lstm_model = tf.keras.models.load_model(lstm_file)
            print("✅ LSTM загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки LSTM: {e}")
    else:
        print("❌ LSTM модель не найдена")

    # ---------- Загрузка GRU ----------
    gru_model = None
    if gru_file:
        print(f"✅ Найден GRU: {gru_file}")
        try:
            gru_model = tf.keras.models.load_model(gru_file)
            print("✅ GRU загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки GRU: {e}")
    else:
        print("❌ GRU модель не найдена")

    # ---------- Оценка ----------
    results = {}
    if tkan_model:
        results['tkan'] = evaluate_model(tkan_model, X_test, y_test, train_max, "TKAN")
    if lstm_model:
        results['lstm'] = evaluate_model(lstm_model, X_test, y_test, train_max, "LSTM")
    if gru_model:
        results['gru'] = evaluate_model(gru_model, X_test, y_test, train_max, "GRU")

    # ---------- Сравнение ----------
    if len(results) >= 2:
        print("\n" + "="*60)
        print("СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ")
        print("="*60)
        comp_data = []
        for key, res in results.items():
            comp_data.append({
                'Model': key.upper(),
                'MAE': res['mae'],
                'RMSE': res['rmse'],
                'R²': res['r2'],
                'MAPE(%)': res['mape']
            })
        comp_df = pd.DataFrame(comp_data)
        print(comp_df.to_string(index=False))

        # Лучшая модель по каждой метрике
        print("\n" + "="*60)
        print("ЛУЧШАЯ МОДЕЛЬ ПО МЕТРИКАМ")
        print("="*60)
        for metric in ['mae', 'rmse', 'mape']:
            best = min(results.keys(), key=lambda k: results[k][metric])
            print(f"✓ {metric.upper()}: {best.upper()} ({results[best][metric]:.6f})")
        best_r2 = max(results.keys(), key=lambda k: results[k]['r2'])
        print(f"✓ R²:      {best_r2.upper()} ({results[best_r2]['r2']:.6f})")

        # Сохраняем результаты
        comp_df.to_csv('sp500_model_comparison.csv', index=False)
        print("\n✅ Сравнение сохранено в 'sp500_model_comparison.csv'")

        # Почасовая таблица (по дням)
        per_day_rows = []
        for d in range(STEPS_AHEAD):
            row = {'Day': d+1}
            for key, res in results.items():
                row[f'{key.upper()}_MAE'] = res['per_day_metrics'][d]['mae']
                row[f'{key.upper()}_R2']  = res['per_day_metrics'][d]['r2']
            per_day_rows.append(row)
        pd.DataFrame(per_day_rows).to_csv('sp500_per_day_comparison.csv', index=False)
        print("✅ Поминутное сравнение (по дням) сохранено в 'sp500_per_day_comparison.csv'")

        # Визуализация
        plot_model_comparison(
            results.get('tkan'),
            results.get('lstm'),
            results.get('gru')
        )

    elif len(results) == 1:
        print("\n⚠️ Загружена только одна модель. Вывожу её результаты.")
        key = list(results.keys())[0]
        res = results[key]
        pd.DataFrame({
            'Metric': ['MAE','RMSE','R²','MAPE'],
            'Value': [res['mae'], res['rmse'], res['r2'], res['mape']]
        }).to_csv(f'sp500_{key}_results.csv', index=False)
        plot_model_comparison(
            results.get('tkan'),
            results.get('lstm'),
            results.get('gru')
        )
    else:
        print("\n❌ Не найдено ни одной модели. Убедитесь, что файлы .keras присутствуют.")

    print("\n" + "="*60)
    print("ОЦЕНКА ЗАВЕРШЕНА")
    print("="*60)