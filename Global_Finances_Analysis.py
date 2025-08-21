import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as d
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional

# -------------------------------
# Page / App Config
# -------------------------------
st.set_page_config(page_title="Global Finance ARIMA Dashboard", layout="wide", page_icon="📈")
st.title("🌍 Global Finance Analysis & ARIMA Forecasts")
st.caption("Markets covered: India, Tokyo, Shanghai, London, Hong Kong • Data: Yahoo Finance")

# -------------------------------
# User Controls (Sidebar)
# -------------------------------
st.sidebar.header("⚙️ Controls")
min_date = d.date(2020, 1, 1)
def_date_start = d.date(2025, 1, 1)
def_date_end = d.date(2025, 7, 10)

start_date = st.sidebar.date_input("Start date", value=def_date_start, min_value=min_date, max_value=d.date.today())
end_date = st.sidebar.date_input("End date", value=def_date_end, min_value=min_date, max_value=d.date.today())
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

forecast_steps = st.sidebar.slider("Forecast horizon (business days)", min_value=5, max_value=30, value=10, step=1)
arima_order = st.sidebar.text_input("ARIMA order (p,d,q)", value="5,1,0")
try:
    p, d_ord, q = [int(x.strip()) for x in arima_order.split(",")]
    arima_order_tuple = (p, d_ord, q)
except Exception:
    st.sidebar.warning("Invalid ARIMA order. Falling back to (5,1,0)")
    arima_order_tuple = (5, 1, 0)

# -------------------------------
# Symbols: Indices & Stocks
# -------------------------------
MARKET_INDEX: Dict[str, str] = {
    "India": "^NSEI",        # NIFTY 50
    "Tokyo": "^N225",        # Nikkei 225
    "Shanghai": "000001.SS",  # SSE Composite Index
    "London": "^FTSE",       # FTSE 100
    "Hong Kong": "^HSI",     # Hang Seng Index
}

INDIA_STOCKS: Dict[str, str] = {
    "TCS": "TCS.NS",
    "Wipro": "WIPRO.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
}

selected_markets = st.sidebar.multiselect("Markets", list(MARKET_INDEX.keys()), default=list(MARKET_INDEX.keys()))
selected_stocks = st.sidebar.multiselect("India Stocks", list(INDIA_STOCKS.keys()), default=["TCS", "Infosys", "HDFC Bank", "ICICI Bank"]) 

# -------------------------------
# Helpers
# -------------------------------
@st.cache_data(show_spinner=False)
def load_yf(ticker: str, start: d.date, end: d.date, auto_adjust: bool = False) -> pd.DataFrame:
    df = yf.download(ticker, start=pd.to_datetime(start), end=pd.to_datetime(end), auto_adjust=auto_adjust, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "Date"})
    # Ensure datetime index for charting convenience
    df["Date"] = pd.to_datetime(df["Date"]) 
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic cleaning
    df = df[[c for c in df.columns if c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]].dropna(subset=["Close"])
    df.sort_values("Date", inplace=True)
    # Daily returns
    df["Returns"] = df["Close"].pct_change()
    # Differenced close
    df["Close_Diff"] = df["Close"].diff()
    return df


def adf_summary(series: pd.Series) -> Dict[str, float]:
    series = series.dropna()
    if len(series) < 20:
        return {"ADF Statistic": np.nan, "p-value": np.nan, "Stationary": False}
    result = adfuller(series)
    return {
        "ADF Statistic": float(result[0]),
        "p-value": float(result[1]),
        "Stationary": bool(result[1] <= 0.05),
    }


def fit_arima_and_forecast(series: pd.Series, steps: int, order: Tuple[int, int, int]) -> Optional[pd.DataFrame]:
    series = series.dropna()
    if len(series) < max(30, order[0] + order[2] + 5):
        return None
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        fc = model_fit.forecast(steps=steps)
        last_date = series.index[-1] if isinstance(series.index, pd.DatetimeIndex) else None
        # business days after last observed date
        if last_date is None:
            # fallback if index isn't datetime
            base_date = pd.to_datetime(pd.Timestamp.today().normalize())
        else:
            base_date = pd.to_datetime(last_date)
        future_dates = pd.bdate_range(start=base_date + pd.Timedelta(days=1), periods=steps)
        out = pd.DataFrame({"Date": future_dates, "Forecast": fc.values})
        return out
    except Exception as ex:
        st.warning(f"ARIMA failed: {ex}")
        return None


def plot_price_and_forecast(df: pd.DataFrame, forecast_df: Optional[pd.DataFrame], title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], label="Actual Close")
    if forecast_df is not None and not forecast_df.empty:
        ax.plot(forecast_df["Date"], forecast_df["Forecast"], linestyle="--", label="Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.set_title(title)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)


def basic_kpis(df: pd.DataFrame, label: str):
    col1, col2, col3, col4 = st.columns(4)
    latest_close = df["Close"].iloc[-1]
    first_close = df["Close"].iloc[0]
    ret = (latest_close / first_close - 1.0) if first_close != 0 else np.nan
    vol = df["Returns"].std() * np.sqrt(252)
    col1.metric(f"{label} Close", f"{latest_close:,.2f}")
    col2.metric("Period Return", f"{ret*100:,.2f}%")
    col3.metric("Ann. Volatility", f"{vol*100:,.2f}%")
    col4.metric("Observations", f"{len(df):,}")

# -------------------------------
# Layout: Tabs
# -------------------------------
market_tabs = st.tabs([f"{m} Index" for m in selected_markets] or ["No market selected"]) 

for tab, market in zip(market_tabs, selected_markets):
    with tab:
        ticker = MARKET_INDEX[market]
        idx_df = load_yf(ticker, start_date, end_date, auto_adjust=False)
        if idx_df.empty:
            st.error(f"No data for {market} ({ticker}).")
            continue
        idx_df = add_derived_columns(idx_df)
        st.subheader(f"{market} — {ticker}")
        basic_kpis(idx_df, label=market)

        # Stationarity
        st.markdown("**Stationarity (ADF Test)**")
        s1 = adf_summary(idx_df["Close"]) 
        s2 = adf_summary(idx_df["Close_Diff"]) 
        c1, c2 = st.columns(2)
        with c1:
            st.write("Close Series")
            st.write({"ADF Statistic": round(s1["ADF Statistic"], 4) if pd.notna(s1["ADF Statistic"]) else None,
                      "p-value": round(s1["p-value"], 4) if pd.notna(s1["p-value"]) else None,
                      "Stationary": s1["Stationary"]})
        with c2:
            st.write("Differenced Close")
            st.write({"ADF Statistic": round(s2["ADF Statistic"], 4) if pd.notna(s2["ADF Statistic"]) else None,
                      "p-value": round(s2["p-value"], 4) if pd.notna(s2["p-value"]) else None,
                      "Stationary": s2["Stationary"]})

        # Forecast
        fc_df = fit_arima_and_forecast(idx_df.set_index("Date")["Close"], steps=forecast_steps, order=arima_order_tuple)
        st.subheader("Price & Forecast")
        plot_price_and_forecast(idx_df, fc_df, f"{market} Index Price & {forecast_steps}d Forecast")
        if fc_df is not None:
            st.dataframe(fc_df.set_index("Date"))
            st.download_button("Download index forecast CSV", data=fc_df.to_csv(index=False), file_name=f"{market}_forecast.csv", mime="text/csv")

# -------------------------------
# Stocks Tab (India)
# -------------------------------
st.header("🇮🇳 India — Selected Stocks")
stock_tabs = st.tabs(selected_stocks or ["No stock selected"]) 

for tab, stock_name in zip(stock_tabs, selected_stocks):
    with tab:
        ticker = INDIA_STOCKS[stock_name]
        df = load_yf(ticker, start_date, end_date, auto_adjust=False)
        if df.empty:
            st.error(f"No data for {stock_name} ({ticker}).")
            continue
        df = add_derived_columns(df)
        st.subheader(f"{stock_name} — {ticker}")
        basic_kpis(df, stock_name)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ADF: Close**")
            s = adf_summary(df["Close"]) 
            st.write({"ADF Statistic": round(s["ADF Statistic"], 4) if pd.notna(s["ADF Statistic"]) else None,
                      "p-value": round(s["p-value"], 4) if pd.notna(s["p-value"]) else None,
                      "Stationary": s["Stationary"]})
        with c2:
            st.markdown("**ADF: Differenced Close**")
            s = adf_summary(df["Close_Diff"]) 
            st.write({"ADF Statistic": round(s["ADF Statistic"], 4) if pd.notna(s["ADF Statistic"]) else None,
                      "p-value": round(s["p-value"], 4) if pd.notna(s["p-value"]) else None,
                      "Stationary": s["Stationary"]})

        fc_df = fit_arima_and_forecast(df.set_index("Date")["Close"], steps=forecast_steps, order=arima_order_tuple)
        st.subheader("Price & Forecast")
        plot_price_and_forecast(df, fc_df, f"{stock_name} Price & {forecast_steps}d Forecast")
        if fc_df is not None:
            st.dataframe(fc_df.set_index("Date"))
            st.download_button("Download stock forecast CSV", data=fc_df.to_csv(index=False), file_name=f"{stock_name}_forecast.csv", mime="text/csv")

# -------------------------------
# Cross-Market Comparison
# -------------------------------
st.header("📊 Cross-Market Comparison")

# Load selected index closes
cmp_frames = []
for m in selected_markets:
    tkr = MARKET_INDEX[m]
    df = load_yf(tkr, start_date, end_date, auto_adjust=False)
    if df.empty:
        continue
    df = add_derived_columns(df)
    cmp_frames.append(df[["Date", "Close"]].assign(Market=m))

if cmp_frames:
    cmp_df = pd.concat(cmp_frames, ignore_index=True)
    piv = cmp_df.pivot(index="Date", columns="Market", values="Close").dropna(how="all")

    st.subheader("Normalized Performance (Start = 100)")
    norm = piv / piv.iloc[0] * 100.0
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in norm.columns:
        ax.plot(norm.index, norm[c], label=c)
    ax.set_title("Index Performance (Normalized)"); ax.set_xlabel("Date"); ax.set_ylabel("Index Level (Start=100)")
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Correlation of Daily Returns")
    rets = piv.pct_change().dropna()
    corr = rets.corr()
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(corr.values)
    ax2.set_xticks(range(len(corr.columns))); ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax2.set_yticks(range(len(corr.index))); ax2.set_yticklabels(corr.index)
    ax2.set_title("Correlation Matrix (Daily Returns)")
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax2.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
    st.pyplot(fig2, clear_figure=True)

    st.download_button("Download normalized performance CSV", data=norm.reset_index().to_csv(index=False), file_name="normalized_performance.csv", mime="text/csv")
else:
    st.info("Select at least one market to view the comparison.")

st.caption("Tip: If ADF indicates non-stationarity for Close, use differencing (d>0) in ARIMA order—already handled by default (d=1). You can experiment with p and q from the sidebar.")
