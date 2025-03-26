

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
df = pd.read_csv(r"C:\Users\priya\OneDrive\Desktop\PRIYA\PROJECT\Nvidia\nvidia_stock_prices.csv")

df.info()

df.head()
df.columns


df.describe()
df.shape

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# EDA 

# Check for missing values and duplicates
print("Missing values:")

print(df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# Summary statistics
print(df.describe())


# Data Visualization 
# Plot stock price trends
plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Close Price')
plt.title('NVIDIA Stock closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# Histogram of closing prices
plt.figure(figsize=(10,5))
sns.histplot(df['Close'], bins=50, kde=True)
plt.title('Distribution of Closing Prices')
plt.show()


# Moving Averages
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Close Price', alpha=0.5)
plt.plot(df['SMA_20'], label='20-day SMA', linestyle='dashed')
plt.plot(df['SMA_50'], label='50-day SMA', linestyle='dashed')
plt.plot(df['SMA_200'], label='200-day SMA', linestyle='dashed')
plt.title('Stock Price with Moving Averages')
plt.legend()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Volatility Analysis
df['Daily Returns'] = df['Close'].pct_change()
plt.figure(figsize=(14,6))
plt.plot(df['Daily Returns'], label='Daily Returns')
plt.title('NVIDIA Stock Daily Returns')
plt.legend()
plt.show()

# Rolling Volatility
df['Rolling Volatility'] = df['Daily Returns'].rolling(window=30).std()
plt.figure(figsize=(14,6))
plt.plot(df['Rolling Volatility'], label='Rolling Volatility (30 days)')
plt.title('Stock Price Volatility Over Time')
plt.legend()
plt.show()

df['Rolling_Volatility'] = df['Close'].rolling(window=30).std()
df[['Close', 'Rolling_Volatility']].plot(subplots=True, figsize=(12,6))
plt.show()


# Seasonality Analysis
result = seasonal_decompose(df['Close'], model='multiplicative', period=252)
result.plot()
plt.show()

# Autocorrelation & Partial Autocorrelation
plt.figure(figsize=(14,6))
plot_acf(df['Close'].dropna(), lags=50)
plt.show()

plt.figure(figsize=(14,6))
plot_pacf(df['Close'].dropna(), lags=50)
plt.show()

"""
saving data for dashboard

import pandas as pd

# Save as CSV (comma-separated values)
df.to_csv('nvidia_dashboard.csv', index=False)  # 'index=False' removes the extra index column

"""

# MODEL BUILDING

df.columns

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = df # Features
y = df['Close']  # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
importances = rf.feature_importances_
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances, color='royalblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance (Random Forest)")
plt.show()

"""


data = df[['Close']] 


# Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create training dataset
train_data = scaled_data[0:int(len(scaled_data) * 0.8)]
X_train = []
y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])


# MODEL BUILDING

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Model Training: 
# Building LSTM model

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Prepare the test dataset (using the last 20% of the data)
test_data = scaled_data[int(len(scaled_data) * 0.8):]
inputs = test_data.reshape(-1, 1)

# Prepare the input data for testing
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Prepare the dates for the predictions
# Since we have a look-back of 60 days, we need to adjust the dates accordingly
predicted_dates = data.index[int(len(data) * 0.8) + 60: int(len(data) * 0.8) + 60 + len(predictions)]

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(data.index[int(len(data) * 0.8):], data['Close'][int(len(data) * 0.8):], color='blue', label='Actual NVDA Stock Price')
plt.plot(predicted_dates, predictions, color='red', label='Predicted Nvidia Stock Price')
plt.title('Nvidia Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

