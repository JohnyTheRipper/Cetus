Cetus - Transformer-based stock price predictor with Walk-Forward Backtesting,
more indicators, overfitting checks, and a final plot of actual vs. predicted vs. MSE.

Key Steps:
1) Asks for ticker, picks a 1-year date range (yesterday minus 365 days).
2) Fetches data from Yahoo Finance and adds multiple indicators:
   - SMA(10), RSI(14), MACD, Bollinger Bands, Stochastic, OBV
3) Creates 60-day sequences for next-day prediction.
4) Walk-forward backtest with n folds (e.g. 5).
   For each fold:
     - Train a fresh model (monitor training MSE).
     - Predict on test data, measure test MSE.
     - Store daily predictions, actuals, and MSE for plotting.
5) Plot:
   - Actual vs. predicted price line graph.
   - Daily MSE line graph.
6) Suggestions on overfitting tweaks.

You can mitigate overfitting by:
 - Using more data (beyond 1 year).
 - Reducing number of indicators (fewer features).
 - Increasing dropout or using smaller d_model, fewer layers.
 - Using early stopping (monitor validation or test performance).
 - Avoiding data leakage (careful with future data usage).
