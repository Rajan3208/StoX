import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import streamlit as st

start = '2015-01-01'
end = '2025-12-31'

st.title('Razzle')
st.subheader('AI based Stock Analysis & Prediction')

stock = st.text_input('Enter Stock Ticker', 'AAPL')

df = yf.download(stock, start=start, end=end)
print(df.head())

# Describing data
st.subheader('Data from 2015-2025')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.title('Closing Price vs Time Chart')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, label='100MA', color='red')
plt.legend()
plt.title('Closing Price vs Time Chart with 100MA')
st.pyplot(fig2)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma200 = df['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, label='100MA', color='red')
plt.plot(ma200, label='200MA', color='green')
plt.title('Closing Price vs Time Chart with 100MA and 200MA')
plt.legend()
st.pyplot(fig3)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load the model
model = load_model('my_model.keras')

# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final graph
st.subheader('Predictions vs Original')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title('Comparison of Original and Predicted Prices')
st.pyplot(fig4)
