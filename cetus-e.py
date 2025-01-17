#!/usr/bin/env python3

"""
Cetus Enhanced - Transformer-based stock price predictor with:
 1) Structured Logging
 2) Modularized Indicator Calculation
 3) Optuna Hyperparameter Optimization (Optional)
 4) Optimized Walk-Forward Backtesting

Usage:
  python cetus_enhanced.py

Key Stages:
 - Logging Setup
 - Indicators (modular functions + main aggregator)
 - Transformer-based model
 - (Optional) Optuna hyperparameter search
 - Optimized walk-forward backtest
 - Final next-day prediction

Requires:
 - Python 3.7+
 - pandas, numpy, matplotlib, yfinance, scikit-learn, tensorflow
 - optuna (for hyperparameter optimization)
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
# 2. Indicators - Modular Functions
# --------------------------------------------------------------------------

def calc_sma(df, window=10, price_col='Close'):
    """Simple Moving Average"""
    return df[price_col].rolling(window).mean()

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

def calc_stoch(df, period=14):
    """Stochastic Oscillator %K, %D"""
    highest_high = df['High'].rolling(period).max()
    lowest_low = df['Low'].rolling(period).min()
    stoch_k = (df['Close'] - lowest_low) * 100.0 / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(3).mean()
    return stoch_k, stoch_d

def calc_obv(df):
    """On-Balance Volume"""
    # sign of close diff times volume
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

def add_indicators(df):
    """Aggregate all indicators into the DataFrame"""
    logger.debug("Adding indicators to DataFrame")
    df['SMA_10'] = calc_sma(df, window=10)
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

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# --------------------------------------------------------------------------
# 3. Data Fetch & Sequence Tools
# --------------------------------------------------------------------------

def get_date_range_one_year():
    """Return (start, end) from 1 year ago to yesterday."""
    today = date.today()
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=365*10)
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
# 4. Transformer Model
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
                            num_blocks=4, dropout=0.1):
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
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# --------------------------------------------------------------------------
# 5. Walk-Forward Backtesting (Optimized for Scalability)
# --------------------------------------------------------------------------

def walk_forward_backtest(X, y, date_indices, dates,
                          build_model_fn,
                          n_splits=5, epochs=5, batch_size=32, target_scaler=None):
    """
    'Growing window' walk-forward:
      For fold i, train=[0..train_end], test=[train_end..train_end+fold_size]
    We'll minimize overhead by pre-slicing data once if desired,
    but for clarity, we'll just do slicing in the loop.

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

        # Evaluate
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
# 7. (Optional) Optuna Hyperparameter Optimization
# --------------------------------------------------------------------------

def objective(trial, X, y, date_indices, dates, target_scaler):
    """
    An example objective function for Optuna hyperparam search.
    We'll do a single pass of walk-forward (with smaller splits or epochs)
    to keep it quick, then measure final average MSE.
    """
    # Sample hyperparams
    d_model = trial.suggest_int('d_model', 16, 128, step=16)
    num_heads = trial.suggest_int('num_heads', 1, 4)
    ff_dim = trial.suggest_int('ff_dim', 32, 256, step=32)
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    num_blocks = trial.suggest_int('num_blocks', 1, 4)

    # Build a partial model function
    def build_model_fn():
        return build_transformer_model(
            seq_length=X.shape[1],
            num_features=X.shape[2],
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_blocks=num_blocks,
            dropout=dropout
        )

    # We'll do a small walk-forward with fewer folds or fewer epochs for speed
    small_n_splits = 3
    small_epochs = 5

    # Execute partial walk-forward
    _, _, _, _, fold_test_mses = walk_forward_backtest(
        X, y, date_indices, dates,
        build_model_fn=build_model_fn,
        n_splits=small_n_splits,
        epochs=small_epochs,
        batch_size=32,
        target_scaler=target_scaler
    )
    # Return average MSE as the metric to minimize
    return float(np.mean(fold_test_mses))


# --------------------------------------------------------------------------
# 8. Main Script
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

    # (C) Add indicators
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
        'Stoch_%K', 'Stoch_%D', 'OBV'
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
    do_optuna = input("Run Optuna hyperparameter optimization? (y/N): ").strip().lower()
    if do_optuna == 'y':
        logger.info("Starting Optuna search for hyperparameters...")
        study = optuna.create_study(direction='minimize')
        # We pass partial data or small folds for speed. Feel free to pass entire sets.
        study.optimize(lambda t: objective(t, X, y, date_indices, dates, targ_scaler), n_trials=10)
        best_params = study.best_params
        logger.info("Best hyperparameters found: %s", best_params)
    else:
        best_params = {
            'd_model': 32,
            'num_heads': 2,
            'ff_dim': 64,
            'dropout': 0.1,
            'num_blocks': 2
        }

    # (F) Build final model function with best params
    def build_model_fn():
        return build_transformer_model(
            seq_length=X.shape[1],
            num_features=X.shape[2],
            d_model=best_params['d_model'],
            num_heads=best_params['num_heads'],
            ff_dim=best_params['ff_dim'],
            num_blocks=best_params['num_blocks'],
            dropout=best_params['dropout']
        )

    # (G) Walk-Forward Backtest with final hyperparams
    n_splits = 5
    epochs = 10
    batch_size = 32
    (all_pred, all_actual, all_dates_str,
     fold_train_losses, fold_test_mses) = walk_forward_backtest(
        X, y, date_indices, dates,
        build_model_fn=build_model_fn,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        target_scaler=targ_scaler
    )
    if len(all_pred) == 0:
        logger.warning("Walk-forward returned no predictions.")
        return

    logger.info("Fold Training Losses: %s", fold_train_losses)
    logger.info("Fold Test MSEs: %s", fold_test_mses)
    avg_mse = float(np.mean(fold_test_mses))
    logger.info("Average Test MSE across folds: %.4f", avg_mse)

    # (H) Plot
    plot_backtest_results(all_dates_str, all_actual, all_pred,
                          title=f"Walk-Forward on {ticker} {start_date}->{end_date}")

    # (I) Retrain on ALL data & Predict Next Day
    logger.info("Retraining final model on ALL data for next-day prediction...")
    final_model = build_model_fn()
    final_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    last_window = X[-1].reshape(1, X.shape[1], X.shape[2])
    scaled_pred = final_model.predict(last_window)
    final_price = targ_scaler.inverse_transform(scaled_pred)[0][0]

    logger.info("======================================")
    logger.info(" Overfitting Concerns: If train loss is low and test MSE is high, reduce model size, ")
    logger.info(" or add dropout, or use more data, or reduce # of indicators.")
    logger.info("======================================")
    logger.info(" FINAL Next-Day predicted closing price for %s: %.2f", ticker, final_price)
    logger.info("==== CETUS ENHANCED SCRIPT END ====")


if __name__ == "__main__":
    main()
