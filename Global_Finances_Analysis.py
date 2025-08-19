import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as d
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# --- Set Date Range ---
s = d.datetime(2025, 1, 1)
e = d.datetime(2025, 7, 10)

# --- Function to Check Stationarity ---
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Stationary': result[1] <= 0.05
    }

# --- ARIMA Analysis Function ---
def arima_analysis(stock_symbol, label):
    df = yf.download(stock_symbol, start=s, end=e, auto_adjust=False)
    if df.empty:
        st.error(f"No data found for {label} ({stock_symbol}) in given date range.")
        return None, None

    df = df[['Close']].copy().reset_index()
    df.dropna(inplace=True)

    # Stationarity checks
    stat_close = check_stationarity(df['Close'])
    df['Close_Diff'] = df['Close'].diff()
    stat_diff = check_stationarity(df['Close_Diff'])

    # Fit ARIMA
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast next 10 business days
    forecast = model_fit.forecast(steps=10)
    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=11, freq='B')[1:]
    forecast_df = pd.DataFrame({'Date': future_dates, f'{label}_Forecast': forecast.values})

    return forecast_df, (stat_close, stat_diff, df)

# --- Streamlit UI ---
st.set_page_config(page_title="ARIMA Stock Forecast", layout="wide")
st.title("📈 ARIMA-Based Stock Price Forecasting")
st.markdown("Analyze and forecast stock prices using ARIMA models.")

# Stock selection
stock_dict = {
    'TCS': 'TCS.NS',
    'Wipro': 'WIPRO.NS',
    'Infosys': 'INFY.NS'
}
stock_choice = st.selectbox("Select a Stock:", list(stock_dict.keys()))
symbol = stock_dict[stock_choice]

# Run analysis
forecast_df, results = arima_analysis(symbol, stock_choice)

if forecast_df is not None:
    stat_close, stat_diff, df = results

    st.subheader(f"Stationarity Check Results for {stock_choice}")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Close Series**")
        st.write(f"ADF Statistic: {stat_close['ADF Statistic']:.4f}")
        st.write(f"p-value: {stat_close['p-value']:.4f}")
        st.write("Stationary ✅" if stat_close['Stationary'] else "Not Stationary ❌")

    with col2:
        st.write("**Differenced Close Series**")
        st.write(f"ADF Statistic: {stat_diff['ADF Statistic']:.4f}")
        st.write(f"p-value: {stat_diff['p-value']:.4f}")
        st.write("Stationary ✅" if stat_diff['Stationary'] else "Not Stationary ❌")

    # Line Plot
    st.subheader("📊 Forecast Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['Close'], label='Actual Close')
    ax.plot(forecast_df['Date'], forecast_df[f'{stock_choice}_Forecast'], label='Forecast', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f"{stock_choice} Price Forecast")
    ax.legend()
    st.pyplot(fig)

    # Table
    st.subheader("🔢 Forecasted Values")
    st.dataframe(forecast_df.set_index('Date'))
