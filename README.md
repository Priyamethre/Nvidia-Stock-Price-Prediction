# Nvidia-Stock-Price-Prediction
## Overview
This dataset provides a comprehensive collection of daily stock price data for Nvidia Corporation (NVDA), spanning a 20-year period from January 2, 2004, to January 1, 2024. Nvidia, a global leader in graphics processing units (GPUs) and AI technologies, has experienced exponential growth, particularly in recent years as it became a major player in artificial intelligence, gaming, and autonomous vehicles. This dataset captures key market movements and trends during Nvidia’s significant rise to prominence.

 ## Business Problem
NVIDIA is a major player in the semiconductor and AI industries, and its stock experiences significant volatility. Investors and analysts need insights into stock price movements, trends, and volatility to make informed decisions. The project aims to:
* Identify historical stock price trends and correlations.
* Evaluate price volatility and seasonality.
* Predict future stock prices using machine learning and deep learning models.


## About the Dataset
The dataset contains crucial financial data for Nvidia's stock, including opening, high, low, and closing prices, as well as trading volume for each day in the 20-year period. This data is ideal for conducting a variety of financial analyses, ranging from simple trend observation to complex predictive modeling using machine learning algorithms such as LSTM (Long Short-Term Memory). Traders, financial analysts, and data scientists can use this dataset to backtest trading strategies, develop stock market prediction models, and perform time series analysis on stock price movements.

## Attribute Information
* Date: The date of the stock price record.
* Open: The stock price at the beginning of the trading day.
* High: The highest price Nvidia stock reached during the day.
* Low: The lowest price Nvidia stock reached during the day.
* Close: The stock price at the end of the trading day.
* Volume: The total number of Nvidia shares traded during the day.

**Preprocessing Steps:**
  - Converted 'Date' column to datetime format and set it as the index.
  - Checked for missing values and handled duplicates.
  - Normalized stock prices using MinMaxScaler for deep learning models.

## **Methodology & Approach**  

###  **Exploratory Data Analysis (EDA)**  
- **Stock Trend Analysis:** Line plots to observe stock movement over time.  
- **Distribution Analysis:** Histograms to analyze price distribution.  
- **Moving Averages (SMA 20, 50, 200):** Trend detection.  
- **Volatility Analysis:** Daily returns and rolling volatility visualization.  
- **Correlation Heatmap:** Identifying feature relationships.  
- **Seasonality Analysis:** Using ACF/PACF plots for time-series trends.  

###  **Feature Engineering & Selection**  
- Extracted **trend and seasonality** using seasonal decomposition.  
- Used **Random Forest feature importance** to select key features.  

###  **Machine Learning Model (Random Forest Regression)**  
- Trained **Random Forest Regressor** on historical data.  
- Evaluated feature importance to understand key stock predictors.  

###  **Deep Learning Model (LSTM - Long Short-Term Memory)**  
- Used a **sequence of 60 past stock prices** to predict future prices.  
- Built an **LSTM neural network** with dropout layers to prevent overfitting.  
- Trained on NVIDIA’s stock data and tested on unseen data.  
- Visualized **actual vs. predicted stock prices**.  

###  **Model Evaluation & Predictions**  
- Compared **actual vs. predicted** stock prices.  
- Assessed model performance using visualization techniques.  

## **Results & Insights**  

- **Trend Capture:** The LSTM model successfully follows NVIDIA’s overall stock trend.  
- **Performance During Volatility:** The model struggles with sharp price fluctuations, lagging slightly behind during high-volatility periods.  
- **Smoothing Effect:** Predictions are smoother than actual prices, as LSTM tends to average sudden changes.  
- **Recent Data Performance:** The model captures trends well but underestimates growth in 2023-2024.    

## **Final Outcome**  
* Developed an **LSTM-based predictive model** for NVIDIA’s stock prices.  
* Provided insights into **market trends, volatility, and price movements**.  
* The model can be **improved** by incorporating **sentiment analysis, technical indicators, and macroeconomic factors**.  

## **Future Scope**  
- **Integrating Financial News Sentiment Analysis** to improve stock movement predictions.  
- **Using more advanced deep learning models** (Transformers, Attention Mechanisms).  
- **Applying predictive analytics with ARIMA/LSTM hybrid models** for better forecasting accuracy.  
