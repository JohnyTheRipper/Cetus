#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import MinMaxScaler

from datetime import date, timedelta


# --------------------------------------------------------------------------
# 1. Utilities: Date range, fetching data
# --------------------------------------------------------------------------

def get_date_range_one_year():
    """
    Returns (start_date, end_date) as strings, from one year ago up to yesterday.
    """
    today = date.today()
    end_date = today - timedelta(days=1)   # yesterday
    start_date = end_date - timedelta(days=365*10)  # set to 10 years for more data
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance for a given ticker
    within a specified date range. Flattens multi-index columns if needed.
    Sorts by date ascending.
    """
    print(f"[Cetus] Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df.reset_index(inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[Cetus] Fetched {len(df)} rows. Columns: {df.columns.tolist()}")
    return df


# --------------------------------------------------------------------------
# 2. Adding Multiple Indicators
# --------------------------------------------------------------------------

def add_indicators(df):
    """
    Adds multiple classic indicators:
      - SMA(10), RSI(14)
      - MACD (12, 26, 9)
      - Bollinger Bands (20-day)
      - Stochastic Oscillator (14-day)
      - On-Balance Volume (OBV)
    Drops rows with NaNs (from rolling).
    """
    if df.empty:
        print("[Cetus] DataFrame is empty; skipping indicators.")
        return df

    # 1) SMA_10
    df['SMA_10'] = df['Close'].rolling(10).mean()

    # 2) RSI(14)
    window_length = 14
    close_diff = df['Close'].diff()
    gain = close_diff.clip(lower=0)
    loss = -1 * close_diff.clip(upper=0)
    avg_gain = gain.ewm(com=window_length - 1, min_periods=window_length).mean()
    avg_loss = loss.ewm(com=window_length - 1, min_periods=window_length).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 3) MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 4) Bollinger Bands (20-day)
    df['BB_MA'] = df['Close'].rolling(window=20).mean()
    df['BB_STD'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_MA'] + 2 * df['BB_STD']
    df['BB_Lower'] = df['BB_MA'] - 2 * df['BB_STD']

    # 5) Stochastic Oscillator (14-day)
    period = 14
    df['14_high'] = df['High'].rolling(period).max()
    df['14_low'] = df['Low'].rolling(period).min()
    df['Stoch_%K'] = (df['Close'] - df['14_low']) * 100.0 / (df['14_high'] - df['14_low'])
    df['Stoch_%D'] = df['Stoch_%K'].rolling(3).mean()

    # 6) On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# --------------------------------------------------------------------------
# 3. Preprocessing: Sequences for next-day forecasting
# --------------------------------------------------------------------------

def create_sequences(df, feature_cols, target_col='Close', sequence_length=60):
    """
    Scales data (features & target), then builds rolling windows of length `sequence_length`.
    Returns X, y, plus the scalers for possible inverse transform.
    Also returns date_indices (the row indices in df for each y[i]) and the full date array for reference.
    """
    needed_cols = set(feature_cols + [target_col])
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"[Cetus] Missing columns in DataFrame: {missing}")

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # We'll keep Date aside to track test sample indices
    dates = df['Date'].values  # for plotting test segments

    # Extract feature/target
    feature_data = df[feature_cols].values  # shape: (num_samples, num_features)
    target_data = df[[target_col]].values   # shape: (num_samples, 1)

    # Scale
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(feature_data)
    scaled_target = target_scaler.fit_transform(target_data)

    X, y = [], []
    date_indices = []  # store the corresponding date index for the "predicted day"
    for i in range(sequence_length, len(df)):
        X.append(scaled_features[i-sequence_length:i])  # prev 60 days
        y.append(scaled_target[i])                      # next day
        date_indices.append(i)                           # so we know which day in df this is

    X = np.array(X)
    y = np.array(y)
    date_indices = np.array(date_indices)
    print(f"[Cetus] Sequences created -> X shape: {X.shape}, y shape: {y.shape}")
    return X, y, date_indices, feature_scaler, target_scaler, dates


# --------------------------------------------------------------------------
# 4. Minimal Transformer
# --------------------------------------------------------------------------

class PositionalEncoding(layers.Layer):
    """
    Simple fixed sine/cosine positional encoding for time series.
    """
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
    """
    One Transformer encoder block:
      - LN -> MultiHeadAttention -> residual
      - LN -> Dense(relu) -> Dense -> residual
    """
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(d_model)(x)
    return x + res


def build_transformer_model(seq_length, num_features,
                            d_model=128,   # hidden dimension (typical range: 16-512)
                            num_heads=4,   # number of attention heads (typical range: 1-8)
                            ff_dim=128,     # feed-forward layer dimension (typical range: 64-512)
                            num_blocks=6,  # number of Transformer encoder blocks (typical range: 1-6)
                            dropout=0.1):  # dropout rate (typical range: 0.1-0.5)

    """
    Builds a Transformer model with the specified hyperparameters.
    """
    inputs = Input(shape=(seq_length, num_features))
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(max_len=seq_length, d_model=d_model)(x)

    for _ in range(num_blocks):
        x = transformer_encoder(x, d_model, num_heads, ff_dim, dropout)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# --------------------------------------------------------------------------
# 5. Walk-Forward Backtesting
# --------------------------------------------------------------------------

def walk_forward_backtest(X, y, date_indices, dates,
                          n_splits=5, build_model_fn=None,
                          epochs=5, batch_size=32, target_scaler=None):
    """
    Splits (X, y) into n_splits folds in a time-series manner (growing window).
    For each fold, it trains a fresh model on X[:train_end], then tests on
    the next chunk X[train_end:train_end + fold_size].

    Returns:
      - all_pred (list of predicted *real* prices) across all folds
      - all_actual (list of actual *real* prices) across all folds
      - all_dates_str (list of date strings for each test sample)
      - fold_train_losses (list of final training MSE for each fold)
      - fold_test_mses (list of MSE across each fold's test set)
    """
    if build_model_fn is None or target_scaler is None:
        raise ValueError("Must provide a model-building function and target_scaler.")

    n_samples = len(X)
    fold_size = n_samples // n_splits

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
            break

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[train_end:train_end + fold_size]
        y_test = y[train_end:train_end + fold_size]
        date_test_indices = date_indices[train_end:train_end + fold_size]

        if len(X_test) == 0:
            break

        # Build & train
        model = build_model_fn()
        train_cb = TrainLossCallback()
        model.fit(X_train, y_train,
                  epochs=epochs, batch_size=batch_size,
                  verbose=0, callbacks=[train_cb])

        final_train_loss = train_cb.train_loss if hasattr(train_cb, 'train_loss') else None
        fold_train_losses.append(final_train_loss)

        # Evaluate test MSE
        test_mse = model.evaluate(X_test, y_test, verbose=0)
        fold_test_mses.append(test_mse)

        # Predict
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_actual = target_scaler.inverse_transform(y_test)

        # Save predictions & actual for plotting
        for i, idx in enumerate(date_test_indices):
            all_pred.append(y_pred[i, 0])
            all_actual.append(y_actual[i, 0])
            # Convert date index to actual date string
            all_dates_str.append(str(dates[idx]))

        print(f"[Cetus] Fold {fold_idx+1}/{n_splits}: Train size={len(X_train)}, "
              f"Test size={len(X_test)}, train_loss={final_train_loss:.4f}, test_MSE={test_mse:.4f}")

    return all_pred, all_actual, all_dates_str, fold_train_losses, fold_test_mses


# --------------------------------------------------------------------------
# 6. Plotting
# --------------------------------------------------------------------------

def plot_backtest_results(all_dates_str, all_actual, all_pred, title):
    """
    Plots actual vs. predicted price, and also the daily MSE (point by point).
    """
    # Convert to np arrays for convenience
    all_dates_str = np.array(all_dates_str)
    all_actual = np.array(all_actual)
    all_pred = np.array(all_pred)

    # Compute daily MSE
    daily_mse = (all_actual - all_pred) ** 2

    # We'll plot them on a single figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Subplot 1: Actual vs Predicted
    axes[0].plot(all_dates_str, all_actual, label='Actual Price', color='blue')
    axes[0].plot(all_dates_str, all_pred, label='Predicted Price', color='red')
    axes[0].set_title(f'{title} - Actual vs. Predicted')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)

    # Subplot 2: Daily MSE
    axes[1].plot(all_dates_str, daily_mse, label='Daily MSE', color='green')
    axes[1].set_title('Daily MSE')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('MSE')
    axes[1].legend()
    axes[1].grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# 7. Main Program (single run with larger Transformer)
# --------------------------------------------------------------------------

def main():
    # A) Ticker + Date Range
    ticker = input("[Cetus] Enter stock ticker (e.g. AAPL): ").strip().upper()
    start_date, end_date = get_date_range_one_year()
    print(f"[Cetus] Using date range: {start_date} to {end_date}")

    # B) Fetch Data
    df = fetch_data(ticker, start_date, end_date)
    if df.empty:
        print("[Cetus] No data fetched. Exiting.")
        return

    # Basic check
    if len(df) < 61:
        print(f"[Cetus] Not enough data ({len(df)}) to build 60-day sequences.")
        return

    # C) Add Indicators
    df = add_indicators(df)
    if len(df) < 61:
        print(f"[Cetus] After adding indicators, only {len(df)} rows remain. Exiting.")
        return

    # D) Create Sequences
    feature_cols = [
        'Open', 'High', 'Low', 'Volume',
        'SMA_10', 'RSI_14',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_MA', 'BB_STD', 'BB_Upper', 'BB_Lower',
        'Stoch_%K', 'Stoch_%D', 'OBV'
    ]
    target_col = 'Close'
    sequence_length = 60

    X, y, date_indices, feat_scaler, tgt_scaler, dates = create_sequences(
        df, feature_cols, target_col, sequence_length
    )

    if len(X) < 2:
        print("[Cetus] Not enough sequence samples for any split.")
        return

    # E) Define our build_model_fn for the LARGER model
    def build_model_fn():
        return build_transformer_model(
            seq_length=sequence_length,
            num_features=X.shape[2],
            d_model=128,
            num_heads=4,
            ff_dim=64,
            num_blocks=3,
            dropout=0.1
        )

    # F) Walk-Forward Backtesting (single run)
    print("[Cetus] Starting Walk-Forward Backtest...\n")

    n_splits = 5
    epochs = 10
    batch_size = 32

    all_pred, all_actual, all_dates_str, fold_train_losses, fold_test_mses = walk_forward_backtest(
        X, y, date_indices, dates,
        n_splits=n_splits,
        build_model_fn=build_model_fn,
        epochs=epochs,
        batch_size=batch_size,
        target_scaler=tgt_scaler
    )

    if len(all_pred) == 0:
        print("[Cetus] Walk-Forward backtest returned no predictions.")
        return

    # Print Overfitting Info
    print("\n[Cetus] Fold Training Losses (final epoch):", fold_train_losses)
    print("[Cetus] Fold Test MSEs:", fold_test_mses)
    avg_test_mse = np.mean(fold_test_mses)
    print(f"[Cetus] Average Test MSE across folds: {avg_test_mse:.4f}")

    # G) Plot Actual vs. Predicted + MSE
    plot_title = f"Walk-Forward on {ticker} from {start_date} to {end_date}"
    plot_backtest_results(all_dates_str, all_actual, all_pred, plot_title)

    # H) Final Next-Day Prediction
    print("\n[Cetus] Training on ALL data for final next-day prediction...")
    final_model = build_model_fn()
    final_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # The last window in X corresponds to the last 60 days in your dataset
    last_window = X[-1].reshape(1, sequence_length, X.shape[2])
    scaled_pred = final_model.predict(last_window)
    final_pred_price = tgt_scaler.inverse_transform(scaled_pred)[0][0]

    # Overfitting discussion
    print("""
[INFO] Overfitting Concerns:
1) If training losses are very low but test MSEs are high, it's overfitting.
2) You can mitigate by:
   - Using more data (longer date range).
   - Reducing model size (fewer blocks, smaller d_model).
   - Increasing dropout or using early stopping.
   - Reducing number of indicators (fewer features).
   - Checking for data leakage (accidentally using future data).
""")

    print(f"[Cetus] Final next-day predicted close for {ticker}: {final_pred_price:.2f}")


if __name__ == "__main__":
    main()
