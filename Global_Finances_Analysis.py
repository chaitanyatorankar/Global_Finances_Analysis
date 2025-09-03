import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as d
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# --- Sidebar Settings ---
st.set_page_config(page_title="Global Finance Analysis - ARIMA Forecast", layout="wide")
st.sidebar.header("⚙️ Customize Analysis")

# Date range input (with default values)
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
    min_value=5, max_value=60, value=10, step=5,
    help="Choose how many business days ahead to forecast"
)

# --- Function to Check Stationarity ---
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

    df = df[['Close']].copy().reset_index()
    df.dropna(inplace=True)

    # Stationarity checks
    stat_close = check_stationarity(df['Close'])
    df['Close_Diff'] = df['Close'].diff()
    stat_diff = check_stationarity(df['Close_Diff'])

    # Fit ARIMA model
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast user-selected horizon
    forecast = model_fit.forecast(steps=forecast_horizon)
    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_horizon+1, freq='B')[1:]
    forecast_df = pd.DataFrame({'Date': future_dates, f'{label}_Forecast': forecast.values})

    return forecast_df, (stat_close, stat_diff, df)

# --- Sector-wise Stock Dictionary ---
sector_stocks = {
    "India": {
        "IT": {
            "TCS": "TCS.NS",
            "Wipro": "WIPRO.NS",
            "Infosys": "INFY.NS",
            "HCL Technologies": "HCLTECH.NS",
            "Tech Mahindra": "TECHM.NS"
        },
        "Banking": {
            "HDFC Bank": "HDFCBANK.NS",
            "ICICI Bank": "ICICIBANK.NS"
        }
    },
    "Japan": {
        "IT": {
            "Sony": "6758.T",
            "Nintendo": "7974.T",
            "Hitachi": "6501.T"
        },
        "Banking": {
            "MUFG": "8306.T"
        },
        "Conglomerates": {
            "Toyota": "7203.T",
            "SoftBank": "9984.T"
        }
    },
    "UK": {
        "Energy": {
            "Shell": "SHEL.L"
        },
        "Banking": {
            "Barclays": "BARC.L",
            "HSBC": "HSBA.L"
        },
        "Pharma": {
            "AstraZeneca": "AZN.L"
        },
        "Beverages": {
            "Diageo": "DGE.L"
        },
        "Publishing": {
            "RELX": "REL.L"
        }
    },
    "Hong Kong": {
        "Tech": {
            "Tencent": "0700.HK",
            "Xiaomi": "1810.HK"
        },
        "Banking": {
            "ICBC": "1398.HK",
            "Bank of China": "3988.HK",
            "Agricultural Bank of China": "1288.HK"
        },
        "Telecom": {
            "China Mobile": "0941.HK"
        }
    }
}

# --- Main UI ---
st.title("🌍 Global Finance Analysis with ARIMA Forecasting")
st.markdown("Analyze IT, Banking & Global stock prices and forecast trends using ARIMA models.")

# --- Multi-level Stock Selection ---
region_choice = st.sidebar.selectbox("🌐 Select Region", list(sector_stocks.keys()))
sector_choice = st.sidebar.selectbox("🏢 Select Sector", list(sector_stocks[region_choice].keys()))
stock_choice = st.sidebar.selectbox("📌 Select a Stock", list(sector_stocks[region_choice][sector_choice].keys()))
symbol = sector_stocks[region_choice][sector_choice][stock_choice]

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

    # Forecast Plot
    st.subheader("📊 Forecast Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['Close'], label='Actual Close')
    ax.plot(forecast_df['Date'], forecast_df[f'{stock_choice}_Forecast'],
            label=f'Forecast ({forecast_days} days)', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f"{stock_choice} Price Forecast - Next {forecast_days} Business Days")
    ax.legend()
    st.pyplot(fig)

    # Forecast Table
    st.subheader("🔢 Forecasted Values")
    st.dataframe(forecast_df.set_index('Date'))
