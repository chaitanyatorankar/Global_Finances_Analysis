import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as d
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

# --- Sidebar Settings ---
st.set_page_config(page_title="Global Finance Analysis - ARIMA Forecast", layout="wide")

st.sidebar.header("⚙️ Customize Analysis")

# Date range input
default_start = d.date(2022, 1, 1)
default_end = d.date(2025, 7, 10)

start_date = st.sidebar.date_input("📅 Start Date", default_start, 
                                   min_value=d.date(2015, 1, 1), 
                                   max_value=d.date.today())

end_date = st.sidebar.date_input("📅 End Date", default_end, 
                                 min_value=start_date, 
                                 max_value=d.date.today())

# Forecast horizon selection
forecast_days = st.sidebar.number_input(
    "🔮 Forecast Days (Business Days)", 
    min_value=5, max_value=60, value=10, step=5
)

# --- Function to Check Stationarity ---
from statsmodels.tsa.stattools import adfuller
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Stationary': result[1] <= 0.05
    }

# --- ARIMA Analysis Function ---
def arima_analysis(stock_symbol, label, s, e, forecast_horizon):
    df = yf.download(stock_symbol, start=s, end=e, auto_adjust=False)
    if df.empty:
        st.error(f"No data found for {label} ({stock_symbol}) in given date range.")
        return None, None

    df = df.reset_index().dropna()

    # Stationarity checks
    stat_close = check_stationarity(df['Close'])
    df['Close_Diff'] = df['Close'].diff()
    stat_diff = check_stationarity(df['Close_Diff'])

    # Fit ARIMA model
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=forecast_horizon)
    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_horizon+1, freq='B')[1:]
    forecast_df = pd.DataFrame({'Date': future_dates, f'{label}_Forecast': forecast.values})

    return forecast_df, (stat_close, stat_diff, df)

# --- Main UI ---
st.title("🌍 Global Finance Analysis with ARIMA Forecasting (Candlesticks + Volume)")
st.markdown("Analyze IT & Banking stock prices and forecast trends using ARIMA models with candlestick charts.")

# --- Stock Selection ---
stock_dict = {
    # IT Companies
    'TCS': 'TCS.NS',
    'Wipro': 'WIPRO.NS',
    'Infosys': 'INFY.NS',
    'HCL Technologies': 'HCLTECH.NS',
    'Tech Mahindra': 'TECHM.NS',

    # Banks
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS'
}

stock_choice = st.selectbox("📌 Select a Stock:", list(stock_dict.keys()))
symbol = stock_dict[stock_choice]

# --- Run Analysis ---
forecast_df, results = arima_analysis(symbol, stock_choice, start_date, end_date, forecast_days)

if forecast_df is not None:
    stat_close, stat_diff, df = results

    # Stationarity results
    st.subheader(f"Stationarity Check Results for {stock_choice}")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Close Series**")
        st.write(f"ADF Statistic: {stat_close['ADF Statistic']:.4f}")
        st.write(f"p-value: {stat_close['p-value']:.4f}")
        st.write("✅ Stationary" if stat_close['Stationary'] else "❌ Not Stationary")

    with col2:
        st.write("**Differenced Close Series**")
        st.write(f"ADF Statistic: {stat_diff['ADF Statistic']:.4f}")
        st.write(f"p-value: {stat_diff['p-value']:.4f}")
        st.write("✅ Stationary" if stat_diff['Stationary'] else "❌ Not Stationary")

    # --- Candlestick + Forecast ---
    st.subheader(f"📊 Candlestick Forecast for {stock_choice} ({forecast_days} Business Days Ahead)")

    fig = go.Figure()

    # Candlestick for actual data
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Actual"
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df[f'{stock_choice}_Forecast'],
        mode='lines+markers',
        line=dict(color="blue", dash="dot"),
        name="Forecast"
    ))

    # Volume bars
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name="Volume",
        marker=dict(color="rgba(128,128,128,0.3)"),
        yaxis="y2"
    ))

    # Layout
    fig.update_layout(
        title=f"{stock_choice} Price Forecast with Volume",
        xaxis=dict(title="Date", rangeslider=dict(visible=False)),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast Table
    st.subheader("🔢 Forecasted Values")
    st.dataframe(forecast_df.set_index('Date'))
