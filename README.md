# Cetus Enhanced: Transformer-Based Stock Price Predictor

![Cetus Logo](https://your-repo-url.com/logo.png) <!-- Replace with your logo or remove if not applicable -->

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [1. Data Fetching](#1-data-fetching)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Sequence Creation](#3-sequence-creation)
  - [4. Model Training](#4-model-training)
    - [a. Transformer Model](#a-transformer-model)
    - [b. LSTM Model](#b-lstm-model)
    - [c. Ensembling](#c-ensembling)
  - [5. Hyperparameter Optimization](#5-hyperparameter-optimization)
  - [6. Walk-Forward Backtesting](#6-walk-forward-backtesting)
  - [7. Evaluation Metrics](#7-evaluation-metrics)
  - [8. Final Prediction](#8-final-prediction)
- [Results](#results)
- [Concerns & Disclaimers](#concerns--disclaimers)
- [Next Steps](#next-steps)
- [License](#license)
- [Contact](#contact)

## Overview

**Cetus Enhanced** is a sophisticated Python-based tool designed for **predicting stock closing prices** using advanced machine learning techniques. Leveraging the power of **Transformer** and **LSTM** architectures, combined with **Optuna** for hyperparameter optimization, Cetus Enhanced aims to deliver high-accuracy predictions through a comprehensive workflow that includes data fetching, feature engineering, model training, backtesting, and final forecasting.

## Features

- **Structured Logging**: Detailed and organized logging for easy debugging and monitoring.
- **Modular Feature Engineering**: Customizable and extendable technical indicators.
- **Hyperparameter Optimization**: Automated tuning using Optuna to enhance model performance.
- **Advanced Models**: Integration of Transformer and LSTM architectures for robust predictions.
- **Ensembling Techniques**: Combines multiple models to improve accuracy and generalization.
- **Walk-Forward Backtesting**: Reliable validation strategy to assess model performance over time.
- **Flexible Evaluation**: Option to visualize results or obtain concise performance metrics.
- **User-Friendly Interface**: Simple prompts for seamless interaction and result interpretation.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/cetus-enhanced.git
   ```

2. **Create a Virtual Environment**

   ```bash
   python3.8 -m venv py38
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   Ensure you have `pip` installed, then run:

   ```bash
   cd cetus-enhanced
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Script**

   ```bash
   python cetus_enhanced.py
   ```

2. **Follow the Prompts**

   - **Enter Stock Ticker**: Provide the ticker symbol (e.g., `AAPL` for Apple Inc.).
   - **Optuna Hyperparameter Optimization**: Choose whether to run hyperparameter tuning (`y/n`).
           (This can take a while, However it drastically improves predictions.)
   - **Display Graph**: Decide if you want to visualize the backtest results with graphs (`y/n`).

3. **View Results**

   - If you choose to display the graph, a detailed visualization of actual vs. predicted prices and daily MSE will appear.
   - If not, you'll receive a summary indicating the percentage of predictions within a 3% error margin.

## How It Works

### 1. Data Fetching

The script fetches historical stock data for the specified ticker over the past year using the **Yahoo Finance API** via the `yfinance` library.

### 2. Feature Engineering

A suite of **technical indicators** is calculated to enrich the dataset:

- **Basic Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator, OBV.
- **Advanced Indicators**: ATR, VWAP, Momentum, Lagged Close Prices.

### 3. Sequence Creation

The enriched data is transformed into sequences suitable for time-series forecasting. Each sequence consists of 60 days of features used to predict the next day's closing price.

### 4. Model Training

#### a. Transformer Model

A Transformer-based neural network processes the sequences to capture complex temporal dependencies.

#### b. LSTM Model

An LSTM-based neural network complements the Transformer model by effectively modeling sequential data.

#### c. Ensembling

Predictions from both the Transformer and LSTM models are averaged to form the final ensemble prediction, enhancing accuracy and robustness.

### 5. Hyperparameter Optimization

Using **Optuna**, the script explores various hyperparameter configurations to identify the optimal settings that minimize the Mean Squared Error (MSE) during validation.

### 6. Walk-Forward Backtesting

A **growing window** walk-forward validation strategy is employed:

- **Training Window**: Expands with each fold to include more data.
- **Testing Window**: Moves forward to validate predictions on unseen data.

### 7. Evaluation Metrics

After backtesting, the script offers two evaluation options:

- **Graphical Visualization**: Plots of actual vs. predicted prices and daily MSE.
- **Hit Ratio Summary**: Reports the percentage of predictions within a 3% error margin, aiming for a 90% hit ratio.

### 8. Final Prediction

The models are retrained on the entire dataset, and a next-day closing price prediction is generated based on the latest data sequence.

## Results

Upon execution, the script provides:

- **Detailed Logs**: Step-by-step process logs for transparency.
- **Backtest Evaluation**: Either visual plots or a concise hit ratio summary.
- **Final Prediction**: The anticipated next-day closing price for the specified stock.

*Example Output:*

```
The model was within 3.25% of the actual price 90.5% of the time.
FINAL Next-Day predicted closing price for AAPL: 150.25
```

## Concerns & Disclaimers

- **Market Volatility**: The stock market is inherently unpredictable. Past performance does not guarantee future results.
- **Overfitting Risks**: Despite regularization and cross-validation, models may overfit to historical data. Continuous monitoring and retraining are essential.
- **Data Quality**: Ensure the integrity and completeness of the fetched data. Inaccurate or missing data can adversely affect model performance.
- **Execution Time**: Hyperparameter optimization and model training can be time-consuming, especially with larger datasets or complex models.
- **Financial Risks**: Use predictions as part of a broader investment strategy. Do not rely solely on model outputs for financial decisions.

## Next Steps

To further enhance Cetus Enhanced and strive toward higher accuracy:

1. **Expand Hyperparameter Tuning**: Increase the number of Optuna trials or explore additional hyperparameters.
2. **Integrate More Models**: Incorporate other architectures like CNNs or Gradient Boosting Machines into the ensemble.
3. **Advanced Feature Selection**: Implement feature importance analysis to retain only the most impactful indicators.
4. **Real-Time Data Integration**: Adapt the script for real-time data fetching and prediction for live trading scenarios.
5. **Deploy as a Service**: Package the model into a web service or API for scalable and accessible predictions.
6. **Automated Reporting**: Generate automated reports or dashboards to visualize performance metrics over time.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

**Viteli Widner**  
Data Scientist & Developer  
[lucaswidner1@gmail.com](mailto:lucaswidner1@gmail.com) 
| [GitHub](https://github.com/JohnyTheRipper) |

---

***Disclaimer***: This project is for ***educational and informational purposes only***. It is not intended as financial advice. Always conduct your own research or consult with a financial professional before making investment decisions.*


