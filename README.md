# Cetus

**Cetus** is an advanced stock prediction tool designed to leverage historical market data and sentiment analysis to deliver accurate and reliable forecasts. Tailored for both seasoned investors and financial professionals, Cetus employs sophisticated machine learning techniques to analyze and predict market movements, providing actionable insights to inform investment strategies.

## Overview

Cetus integrates a comprehensive suite of technical indicators with sentiment analysis derived from real-time news data to generate precise stock price predictions. Utilizing a combination of Transformer and LSTM-based neural networks, Cetus performs walk-forward backtesting to ensure robustness and adaptability across diverse market conditions. The system's modular architecture allows for seamless customization and optimization, enabling users to fine-tune indicator weights and model hyperparameters to suit specific investment objectives.

Key components of Cetus include:

- **Technical Indicator Integration:** Incorporates a wide range of indicators such as Moving Averages, RSI, MACD, Bollinger Bands, and more to capture various market dynamics.
- **Sentiment Analysis:** Utilizes NewsAPI to fetch and analyze news headlines, gauging market sentiment to enhance prediction accuracy.
- **Hyperparameter Optimization:** Employs Optuna to systematically optimize model parameters and indicator weights, ensuring optimal performance.
- **Walk-Forward Backtesting:** Implements a rigorous backtesting framework to validate model predictions and assess performance over time.
- **Efficient Data Handling:** Designed to process large datasets spanning 5-10 years, ensuring scalability and efficiency without compromising on performance.

## Features

- **Comprehensive Indicator Suite:** Leverages multiple technical indicators to provide a holistic view of market trends and movements.
- **Dynamic Sentiment Integration:** Incorporates real-time news sentiment to adjust predictions based on prevailing market emotions.
- **Advanced Machine Learning Models:** Utilizes Transformer and LSTM architectures for capturing both short-term and long-term dependencies in data.
- **Automated Optimization:** Features integrated Optuna-based optimization for fine-tuning model parameters and indicator weights.
- **Robust Backtesting Framework:** Ensures model reliability through extensive walk-forward backtesting across multiple market scenarios.
- **User-Friendly Interface:** Designed with ease of use in mind, allowing users to interactively adjust settings and visualize results.
- **Scalable Architecture:** Capable of handling extensive historical data efficiently, making it suitable for long-term investment analyses.

## Visual References

![Cetus Workflow](images/cetus_workflow.png)
*Figure 1: High-level workflow of the Cetus prediction system.*

![Model Architecture](images/model_architecture.png)
*Figure 2: Transformer and LSTM model architectures employed in Cetus.*

![Backtesting Results](images/backtesting_results.png)
*Figure 3: Sample backtesting results showcasing model performance over time.*

---

*For more information or inquiries, please contact me at lucaswidner1@gmail.com.*