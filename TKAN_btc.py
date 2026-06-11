import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint


# ==========================================
# 1. Custom KAN Activation Layer
# ==========================================
# ==========================================
# 1. Custom KAN Activation Layer
# ==========================================
class KANBasisLayer(tf.keras.layers.Layer):
    """
    A Kolmogorov-Arnold Network (KAN) sublayer.
    Replaces fixed activations with learnable basis expansions.
    """

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

        # FIXED: Use add_weight with trainable=False instead of tf.constant
        # This ensures the tensor is globally tracked by the Keras state machine
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
# 2. Custom TKAN Recurrent Layer
# ==========================================
class CustomTKAN(tf.keras.layers.Layer):
    """
    A completely native scratch-built implementation of the TKAN layer
    combining RKAN layers with LSTM gating mechanisms.
    """

    def __init__(self, units, return_sequences=False, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.num_layers = num_layers

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # LSTM Gates Weight Matrices
        self.W_f = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="W_f")
        self.U_f = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="U_f")
        self.b_f = self.add_weight(shape=(self.units,), initializer="zeros", name="b_f")

        self.W_i = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="W_i")
        self.U_i = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="U_i")
        self.b_i = self.add_weight(shape=(self.units,), initializer="zeros", name="b_i")

        self.W_c = self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform", name="W_c")
        self.U_c = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="U_c")
        self.b_c = self.add_weight(shape=(self.units,), initializer="zeros", name="b_c")

        # RKAN Hidden State weights
        self.W_x_tilde = [
            self.add_weight(shape=(input_dim, input_dim), initializer="glorot_uniform", name=f"W_x_tilde_{l}") for l in
            range(self.num_layers)]
        self.W_h_tilde = [
            self.add_weight(shape=(self.units, input_dim), initializer="glorot_uniform", name=f"W_h_tilde_{l}") for l in
            range(self.num_layers)]

        self.W_hh = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="W_hh")
        self.W_hz = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", name="W_hz")

        self.kan_layers = [KANBasisLayer(input_dim=input_dim, output_dim=self.units) for _ in range(self.num_layers)]

        # Output Gate Weights
        self.W_o = self.add_weight(shape=(self.units * self.num_layers, self.units), initializer="glorot_uniform",
                                   name="W_o")
        self.b_o = self.add_weight(shape=(self.units,), initializer="zeros", name="b_o")

        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = inputs.shape[1]  # Using static integer dimension to prevent symbolic execution errors

        # Initialize internal memory structures to zeros
        h_t = tf.zeros((batch_size, self.units))
        c_t = tf.zeros((batch_size, self.units))
        h_tilde_layers = [tf.zeros((batch_size, self.units)) for _ in range(self.num_layers)]

        output_sequences = tf.TensorArray(dtype=tf.float32, size=time_steps)

        # Loop chronologically through every timestep in the sequence
        for t in range(time_steps):
            x_t = inputs[:, t, :]

            # 1. Compute Standard LSTM Gates
            f_t = tf.sigmoid(tf.matmul(x_t, self.W_f) + tf.matmul(h_t, self.U_f) + self.b_f)
            i_t = tf.sigmoid(tf.matmul(x_t, self.W_i) + tf.matmul(h_t, self.U_i) + self.b_i)
            c_tilde_t = tf.sigmoid(tf.matmul(x_t, self.W_c) + tf.matmul(h_t, self.U_c) + self.b_c)

            # 2. Process through RKAN Layers
            kan_outputs = []
            for l in range(self.num_layers):
                s_lt = tf.matmul(x_t, self.W_x_tilde[l]) + tf.matmul(h_tilde_layers[l], self.W_h_tilde[l])
                o_tilde_t = self.kan_layers[l](s_lt)
                kan_outputs.append(o_tilde_t)
                h_tilde_layers[l] = tf.matmul(h_tilde_layers[l], self.W_hh) + tf.matmul(o_tilde_t, self.W_hz)

            # 3. Concatenate KAN Layers & Calculate Output Gate
            r_t = tf.concat(kan_outputs, axis=-1)
            o_t = tf.sigmoid(tf.matmul(r_t, self.W_o) + self.b_o)

            # 4. Long-Term Cell State & Final Hidden Output Updates
            c_t = f_t * c_t + i_t * c_tilde_t
            h_t = o_t * tf.tanh(c_t)

            output_sequences = output_sequences.write(t, h_t)

        output_sequences = tf.transpose(output_sequences.stack(), perm=[1, 0, 2])

        if self.return_sequences:
            return output_sequences
        else:
            return output_sequences[:, -1, :]


# ==========================================
# 3. Execution Pipeline
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

    return train_data, test_data


def create_sequences(data, feature_cols, target_col):
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH - STEPS_AHEAD + 1):
        X.append(data[feature_cols].iloc[i: i + SEQ_LENGTH].values)
        y.append(data[target_col].iloc[i + SEQ_LENGTH: i + SEQ_LENGTH + STEPS_AHEAD].values)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    csv_file_path = 'btc.csv'

    # ИСПРАВЛЕНИЕ: Распаковываем только 2 переменные
    train_df, test_df = prepare_and_scale_data(csv_file_path)

    features = ['target_final', 'hour_of_day', 'day_of_week']

    X_train, y_train = create_sequences(train_df, features, 'target_final')
    X_test, y_test = create_sequences(test_df, features, 'target_final')

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        CustomTKAN(units=100, return_sequences=True, num_layers=2),
        CustomTKAN(units=100, return_sequences=False, num_layers=2),
        Dense(STEPS_AHEAD, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # НОВЫЙ КОД: Добавляем ModelCheckpoint для страховки
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        # Эта строчка будет сохранять модель в файл 'best_model.keras' каждый раз,
        # когда метрика val_loss улучшается.
        ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print(f"Data shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}")
    print("Beginning training sequence...")

    model.fit(
        X_train, y_train,
        validation_split=0.20,
        epochs=15,
        batch_size=64,
        callbacks=callbacks
    )

    print("\nTraining complete. Evaluating on test window:")
    model.evaluate(X_test, y_test)

    # Финальное сохранение на всякий случай
    model.save('final_tkan_model.keras')
    print("Модель успешно сохранена на диск!")


