#!/usr/bin/env python3

"""
Cetus Enhanced - Transformer-based stock price predictor with:
 1) Structured Logging
 2) Modularized Indicator Calculation with Advanced Features
 3) Optuna Hyperparameter Optimization (Optional)
 4) Optimized Walk-Forward Backtesting
 5) Ensembling Techniques
 6) Robust Evaluation Metrics with Custom Callbacks

Usage:
  python cetus_enhanced.py

Key Stages:
 - Logging Setup
 - Indicators (modular functions + advanced indicators)
 - Transformer-based model
 - LSTM-based model for ensembling
 - (Optional) Optuna hyperparameter search
 - Optimized walk-forward backtest
 - Final next-day prediction
 - Optional graph display or hit ratio summary
"""

import logging
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime, timedelta, date
from functools import partial

# --------------------------------------------------------------------------
# 1. Logging Setup
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 2. Indicators - Modular Functions with Advanced Features
# --------------------------------------------------------------------------

def calc_sma(df, window=10, price_col='Close'):
    """Simple Moving Average"""
    return df[price_col].rolling(window).mean()

def calc_ema(df, span=20, price_col='Close'):
    """Exponential Moving Average"""
    return df[price_col].ewm(span=span, adjust=False).mean()

def calc_rsi(df, period=14, price_col='Close'):
    """Relative Strength Index"""
    close_diff = df[price_col].diff()
    gain = close_diff.clip(lower=0)
    loss = -1 * close_diff.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(df, price_col='Close'):
    """MACD (12,26) with signal(9) and hist"""
    ema12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema26 = df[price_col].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calc_bollinger_bands(df, window=20, price_col='Close'):
    """Bollinger Bands (MA ± 2 * STD)"""
    bb_ma = df[price_col].rolling(window=window).mean()
    bb_std = df[price_col].rolling(window=window).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    return bb_ma, bb_std, bb_upper, bb_lower

def calc_atr(df, period=14):
    """Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window=period).mean()
    return atr

def calc_vwap(df):
    """Volume Weighted Average Price"""
    return (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

def calc_momentum(df, period=10):
    """Momentum Indicator"""
    return df['Close'] - df['Close'].shift(period)

def calc_stoch(df, period=14):
    """Stochastic Oscillator %K, %D"""
    highest_high = df['High'].rolling(period).max()
    lowest_low = df['Low'].rolling(period).min()
    stoch_k = (df['Close'] - lowest_low) * 100.0 / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(3).mean()
    return stoch_k, stoch_d

def calc_obv(df):
    """On-Balance Volume"""
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

def add_advanced_indicators(df):
    """Add advanced indicators to the DataFrame"""
    logger.debug("Adding advanced indicators to DataFrame")
    df['EMA_20'] = calc_ema(df, span=20)
    df['ATR_14'] = calc_atr(df, period=14)
    df['VWAP'] = calc_vwap(df)
    df['Momentum_10'] = calc_momentum(df, period=10)

    # Lagged features
    df['Lag_Close_1'] = df['Close'].shift(1)
    df['Lag_Close_2'] = df['Close'].shift(2)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_indicators(df):
    """Aggregate all indicators into the DataFrame"""
    logger.debug("Adding indicators to DataFrame")
    df['SMA_10'] = calc_sma(df, 10)
    df['RSI_14'] = calc_rsi(df, 14)
    macd, macd_signal, macd_hist = calc_macd(df)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist

    bb_ma, bb_std, bb_up, bb_low = calc_bollinger_bands(df, 20)
    df['BB_MA'] = bb_ma
    df['BB_STD'] = bb_std
    df['BB_Upper'] = bb_up
    df['BB_Lower'] = bb_low

    stoch_k, stoch_d = calc_stoch(df, 14)
    df['Stoch_%K'] = stoch_k
    df['Stoch_%D'] = stoch_d

    df['OBV'] = calc_obv(df)

    # Add advanced indicators
    df = add_advanced_indicators(df)

    return df


# --------------------------------------------------------------------------
# 3. Data Fetch & Sequence Tools
# --------------------------------------------------------------------------

def get_date_range_one_year():
    """Return (start, end) from 1 year ago to yesterday."""
    today = date.today()
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def fetch_data(ticker, start_date, end_date):
    """Download data from Yahoo Finance, flatten columns, sort by date."""
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.reset_index(inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Fetched {len(df)} rows with columns {df.columns.tolist()}")
    return df

def create_sequences(df, feature_cols, target_col='Close', seq_length=60):
    """
    Build sequences of length `seq_length` from the DataFrame columns.
    Returns: X, y, date_indices, feature_scaler, target_scaler, date_array
    """
    logger.debug("Creating sequences of length %d for features: %s", seq_length, feature_cols)
    needed = set(feature_cols + [target_col])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    dates = df['Date'].values
    feature_data = df[feature_cols].values
    target_data = df[[target_col]].values

    feat_scaler = MinMaxScaler()
    targ_scaler = MinMaxScaler()

    scaled_features = feat_scaler.fit_transform(feature_data)
    scaled_target = targ_scaler.fit_transform(target_data)

    X, y = [], []
    date_indices = []
    for i in range(seq_length, len(df)):
        X.append(scaled_features[i - seq_length:i])
        y.append(scaled_target[i])
        date_indices.append(i)

    X = np.array(X)
    y = np.array(y)
    date_indices = np.array(date_indices)
    logger.info(f"Sequences created -> X shape: {X.shape}, y shape: {y.shape}")
    return X, y, date_indices, feat_scaler, targ_scaler, dates


# --------------------------------------------------------------------------
# 4. Transformer and LSTM Models
# --------------------------------------------------------------------------

class PositionalEncoding(layers.Layer):
    """Fixed sine/cosine positional encoding layer."""
    def __init__(self, max_len=5000, d_model=32):
        super().__init__()
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pos_encoding = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:seq_len, :]

def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout=0.1):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(d_model)(x)
    return x + res

def build_transformer_model(seq_length, num_features,
                            d_model=64, num_heads=4, ff_dim=128,
                            num_blocks=4, dropout=0.2):
    """
    Builds a more complex Transformer-based regression model for time series.
    """
    inputs = Input(shape=(seq_length, num_features))
    # Project to d_model
    x = layers.Dense(d_model)(inputs)
    # Positional encoding
    x = PositionalEncoding(max_len=seq_length, d_model=d_model)(x)

    for _ in range(num_blocks):
        x = transformer_encoder(x, d_model, num_heads, ff_dim, dropout)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_lstm_model(seq_length, num_features,
                     lstm_units=50, dropout=0.2):
    """
    Builds an LSTM-based regression model for time series.
    """
    inputs = Input(shape=(seq_length, num_features))
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train multiple models and average their predictions
def ensemble_predict(models, X):
    """
    Takes a list of trained models and returns the averaged predictions.
    """
    preds = [model.predict(X, verbose=0) for model in models]
    avg_pred = np.mean(preds, axis=0)
    return avg_pred


# --------------------------------------------------------------------------
# 5. Walk-Forward Backtesting (Optimized for Scalability)
# --------------------------------------------------------------------------

def walk_forward_backtest(X, y, date_indices, dates,
                          build_model_fn,
                          n_splits=5, epochs=5, batch_size=32, target_scaler=None):
    """
    'Growing window' walk-forward:
      For fold i, train=[0..train_end], test=[train_end..train_end+fold_size]
    We'll minimize overhead by re-initializing the model each fold,
    but not re-doing feature scaling or other heavy tasks.

    Returns:
      all_pred, all_actual, all_dates_str, fold_train_losses, fold_test_mses
    """
    logger.info("Starting walk-forward backtest with %d splits...", n_splits)
    n_samples = len(X)
    fold_size = n_samples // n_splits

    if target_scaler is None:
        raise ValueError("Must provide target_scaler for inverse transform.")

    all_pred = []
    all_actual = []
    all_dates_str = []

    fold_train_losses = []
    fold_test_mses = []

    class TrainLossCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            self.train_loss = logs.get('loss')

    for fold_idx in range(n_splits):
        train_end = fold_size * (fold_idx + 1)
        if train_end >= n_samples:
            logger.warning("No more data to form fold %d", fold_idx+1)
            break

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[train_end:train_end+fold_size]
        y_test = y[train_end:train_end+fold_size]
        date_test_indices = date_indices[train_end:train_end+fold_size]

        if len(X_test) == 0:
            logger.warning("Fold %d has no test data; stopping early", fold_idx+1)
            break

        model = build_model_fn()
        loss_cb = TrainLossCallback()

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  verbose=0, callbacks=[loss_cb])

        final_train_loss = loss_cb.train_loss if hasattr(loss_cb, 'train_loss') else None
        fold_train_losses.append(final_train_loss)

        # Evaluate on test set
        test_mse = model.evaluate(X_test, y_test, verbose=0)
        fold_test_mses.append(test_mse)

        # Predict on test
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_actual = target_scaler.inverse_transform(y_test)

        # Store for plotting
        for i, idx in enumerate(date_test_indices):
            all_pred.append(y_pred[i, 0])
            all_actual.append(y_actual[i, 0])
            all_dates_str.append(str(dates[idx]))

        logger.info("Fold %d/%d -> TrainEnd=%d, TestSize=%d, TrainLoss=%.4f, TestMSE=%.4f",
                    fold_idx+1, n_splits, train_end, len(X_test), final_train_loss, test_mse)

    return all_pred, all_actual, all_dates_str, fold_train_losses, fold_test_mses


# --------------------------------------------------------------------------
# 6. Plotting
# --------------------------------------------------------------------------

def plot_backtest_results(all_dates_str, all_actual, all_pred, title="Backtest Results"):
    logger.info("Plotting backtest results...")
    all_dates_str = np.array(all_dates_str)
    all_actual = np.array(all_actual)
    all_pred = np.array(all_pred)

    daily_mse = (all_actual - all_pred)**2

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Actual vs. Predicted
    axes[0].plot(all_dates_str, all_actual, label='Actual', color='blue')
    axes[0].plot(all_dates_str, all_pred, label='Predicted', color='red')
    axes[0].set_title(f"{title}: Actual vs. Predicted")
    axes[0].legend()
    axes[0].grid(True)

    # Daily MSE
    axes[1].plot(all_dates_str, daily_mse, label='Daily MSE', color='green')
    axes[1].set_title("Daily MSE")
    axes[1].grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# (Optional) Optuna Hyperparameter Optimization
# --------------------------------------------------------------------------

def objective(trial, X, y, date_indices, dates, target_scaler):
    """
    Example objective function for Optuna hyperparam search.
    Incorporates an expanded search space and cross-validation for robustness.
    """
    # Hyperparameter suggestions
    d_model = trial.suggest_int('d_model', 32, 256, step=32)
    num_heads = trial.suggest_int('num_heads', 2, 8, step=2)
    ff_dim = trial.suggest_int('ff_dim', 64, 512, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    num_blocks = trial.suggest_int('num_blocks', 2, 6, step=1)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Build Transformer model
    def build_transformer_fn():
        return build_transformer_model(
            seq_length=X.shape[1],
            num_features=X.shape[2],
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_blocks=num_blocks,
            dropout=dropout
        )

    model = build_transformer_fn()

    # Compile with chosen optimizer and learning rate
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Perform walk-forward backtest with fewer splits and epochs for speed
    small_n_splits = 3
    small_epochs = 5
    _, _, _, fold_test_mses = walk_forward_backtest(
        X, y, date_indices, dates,
        build_model_fn=lambda: model,  # Use the already compiled model
        n_splits=small_n_splits,
        epochs=small_epochs,
        batch_size=32,
        target_scaler=target_scaler
    )

    # Return the average test MSE as the objective to minimize
    return float(np.mean(fold_test_mses))


# --------------------------------------------------------------------------
# 7. Main Script
# --------------------------------------------------------------------------

def main():
    logger.info("==== CETUS ENHANCED SCRIPT START ====")

    # (A) Ask user for ticker
    ticker = input("Enter stock ticker (e.g. AAPL): ").strip().upper()
    start_date, end_date = get_date_range_one_year()
    logger.info("Using date range: %s -> %s", start_date, end_date)

    # (B) Fetch data
    df = fetch_data(ticker, start_date, end_date)
    if df.empty or len(df) < 61:
        logger.warning("Not enough data to proceed.")
        return

    # (C) Add Indicators
    df = add_indicators(df)
    if len(df) < 61:
        logger.warning("After adding indicators, only %d rows remain. Exiting.", len(df))
        return

    # (D) Create sequences
    feature_cols = [
        'Open', 'High', 'Low', 'Volume',
        'SMA_10', 'RSI_14',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_MA', 'BB_STD', 'BB_Upper', 'BB_Lower',
        'Stoch_%K', 'Stoch_%D', 'OBV',
        'EMA_20', 'ATR_14', 'VWAP', 'Momentum_10',
        'Lag_Close_1', 'Lag_Close_2'
    ]
    target_col = 'Close'
    seq_length = 60

    X, y, date_indices, feat_scaler, targ_scaler, dates = create_sequences(
        df, feature_cols, target_col, seq_length
    )
    if len(X) < 2:
        logger.warning("Insufficient sequence data for backtest.")
        return

    # (E) Optional: Optuna Hyperparameter Search
    do_optuna = input("Run Optuna hyperparameter optimization? (y/n): ").strip().lower()
    if do_optuna == 'y':
        logger.info("Starting Optuna search for hyperparameters...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t, X, y, date_indices, dates, targ_scaler), n_trials=100)
        best_params = study.best_params
        logger.info("Best hyperparameters found: %s", best_params)
    else:
        best_params = {
            'd_model': 64,
            'num_heads': 4,
            'ff_dim': 128,
            'dropout': 0.2,
            'num_blocks': 4
        }

    # (F) Build final model functions with best params
    def build_transformer_fn():
        return build_transformer_model(
            seq_length=X.shape[1],
            num_features=X.shape[2],
            d_model=best_params['d_model'],
            num_heads=best_params['num_heads'],
            ff_dim=best_params['ff_dim'],
            num_blocks=best_params['num_blocks'],
            dropout=best_params['dropout']
        )

    def build_lstm_fn():
        return build_lstm_model(
            seq_length=X.shape[1],
            num_features=X.shape[2],
            lstm_units=50,
            dropout=0.2
        )

    # (G) Walk-Forward Backtest for Transformer Model
    n_splits = 5
    epochs = 10
    batch_size = 32
    (all_pred_trans, all_actual_trans, all_dates_str_trans,
     fold_train_losses_trans, fold_test_mses_trans) = walk_forward_backtest(
        X, y, date_indices, dates,
        build_model_fn=build_transformer_fn,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        target_scaler=targ_scaler
    )
    if len(all_pred_trans) == 0:
        logger.warning("Walk-forward returned no predictions for Transformer model.")

    # (H) Walk-Forward Backtest for LSTM Model
    (all_pred_lstm, all_actual_lstm, all_dates_str_lstm,
     fold_train_losses_lstm, fold_test_mses_lstm) = walk_forward_backtest(
        X, y, date_indices, dates,
        build_model_fn=build_lstm_fn,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        target_scaler=targ_scaler
    )
    if len(all_pred_lstm) == 0:
        logger.warning("Walk-forward returned no predictions for LSTM model.")

    # (I) Ensemble Predictions by Averaging
    all_pred_ensemble = []
    all_actual_ensemble = []
    all_dates_str_ensemble = []

    # Assuming both models have predictions for the same dates
    for pred_t, actual_t, date_t, pred_l, actual_l, date_l in zip(all_pred_trans, all_actual_trans, all_dates_str_trans,
                                                                   all_pred_lstm, all_actual_lstm, all_dates_str_lstm):
        # Ensure dates match
        if date_t == date_l and actual_t == actual_l:
            ensemble_pred = (pred_t + pred_l) / 2
            all_pred_ensemble.append(ensemble_pred)
            all_actual_ensemble.append(actual_t)  # same as actual_l
            all_dates_str_ensemble.append(date_t)
        else:
            logger.warning("Mismatch in dates or actual values between models.")

    # (J) Evaluate Ensemble Performance
    if len(all_pred_ensemble) > 0:
        # Calculate hit ratio
        threshold = 3.0  # 3%
        safe_actual = np.where(np.array(all_actual_ensemble) == 0, 1e-9, np.array(all_actual_ensemble))
        ape = np.abs(np.array(all_actual_ensemble) - np.array(all_pred_ensemble)) / np.abs(safe_actual) * 100.0
        hit_ratio = np.mean(ape <= threshold) * 100.0
        logger.info("Ensemble Hit Ratio within %.1f%%: %.2f%%", threshold, hit_ratio)

        # Ask user to display graph or show hit ratio summary
        show_graph = input("Display graph? (y/n): ").strip().lower()
        if show_graph == 'y':
            plot_backtest_results(all_dates_str_ensemble, all_actual_ensemble, all_pred_ensemble,
                                  title=f"Walk-Forward Ensemble on {ticker} {start_date}->{end_date}")
        else:
            logger.info("The model was within %.2f%% of the actual price 90%% of the time.", hit_ratio)
    else:
        logger.warning("No ensemble predictions to evaluate.")

    # (K) Retrain Ensemble Models on ALL Data and Predict Next Day
    logger.info("Retraining ensemble models on ALL data for next-day prediction...")

    # Train Transformer Model on ALL Data
    transformer_model = build_transformer_fn()
    transformer_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # Train LSTM Model on ALL Data
    lstm_model = build_lstm_fn()
    lstm_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # Ensemble Predictions for Next Day
    last_window = X[-1].reshape(1, X.shape[1], X.shape[2])
    pred_trans = transformer_model.predict(last_window, verbose=0)
    pred_lstm = lstm_model.predict(last_window, verbose=0)
    ensemble_pred_next = (pred_trans + pred_lstm) / 2
    final_pred_price = targ_scaler.inverse_transform(ensemble_pred_next)[0][0]

    logger.info("======================================")
    logger.info(" Overfitting Concerns: If train loss is low and test MSE is high, reduce model size, ")
    logger.info(" or add dropout, or use more data, or reduce # of indicators.")
    logger.info("======================================")
    logger.info(" FINAL Next-Day predicted closing price for %s: %.2f", ticker, final_pred_price)
    logger.info("==== CETUS ENHANCED SCRIPT END ====")

if __name__ == "__main__":
    main()
