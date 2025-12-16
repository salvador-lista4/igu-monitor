import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import datetime
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="IGU v5.2 Live Monitor", layout="wide")

@st.cache_data(ttl=86400)
def run_igu_model():
    # === PARÁMETROS ORIGINALES ===
    START, END = "1959-01-01", datetime.datetime.now().strftime("%Y-%m-%d")
    SG_WINDOW, SG_POLY = 11, 3
    TARGET_HORIZON_MONTHS = 12
    
    # === 1) DATOS S&P500 ===
    sp = yf.download("^GSPC", start=START, end=END, progress=False, auto_adjust=False)
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = sp.columns.get_level_values(0)
    price_col = "Adj Close" if "Adj Close" in sp.columns else "Close"
    sp = sp[[price_col]].rename(columns={price_col: "price"})
    sp["ret"] = np.log(sp["price"]).diff()
    sp["rv30"] = sp["ret"].rolling(30, min_periods=10).std() * np.sqrt(252)
    pseudo_monthly = sp["rv30"].resample("M").last().to_frame("pseudo_vix")

    # === 2) DATOS FRED ===
    fred_codes = {"DGS10": "US10Y", "DGS2": "US2Y", "BAMLH0A0HYM2": "HY_spread", "TEDRATE": "TED", "M2SL": "M2"}
    macro_frames = []
    for code, name in fred_codes.items():
        try:
            s = pdr.DataReader(code, "fred", start="1960-01-01", end=END).resample("M").last()
            macro_frames.append(s.rename(columns={code: name}))
        except: pass
    macro_df = pd.concat(macro_frames, axis=1)
    if "US10Y" in macro_df.columns and "US2Y" in macro_df.columns:
        macro_df["curve"] = macro_df["US10Y"] - macro_df["US2Y"]

    # === 3) MERGE E IMPUTACIÓN ===
    df = pseudo_monthly.join(macro_df, how="outer")
    df = df[(df.index >= "1960-01-01")].sort_index()
    df = df.interpolate(limit=3).ffill().bfill()

    # === 4) DERIVADAS SAVITZKY-GOLAY ===
    V = -np.log(np.clip(df["pseudo_vix"].values, 1e-8, None))
    df["V"] = savgol_filter(V, SG_WINDOW, SG_POLY, deriv=0, mode="interp")
    df["dV"] = savgol_filter(V, SG_WINDOW, SG_POLY, deriv=1, delta=1.0, mode="interp")
    df["d2V"] = savgol_filter(V, SG_WINDOW, SG_POLY, deriv=2, delta=1.0, mode="interp")
    df["d3V"] = savgol_filter(V, SG_WINDOW, SG_POLY, deriv=3, delta=1.0, mode="interp")
    df["d3V_s"] = df["d3V"].ewm(span=4).mean()

    # === 5) FEATURE ENGINEERING ===
    df["vix_ma6"] = df["pseudo_vix"].rolling(6).mean()
    df["vix_std12"] = df["pseudo_vix"].rolling(12).std()
    df["M2_yoy"] = df["M2"].pct_change(12).fillna(0)
    monthly_price = sp["price"].resample("M").last().reindex(df.index)
    df["drawdown_12"] = (monthly_price / monthly_price.rolling(12).max() - 1).fillna(0)
    df["ret_kurt_12"] = sp["ret"].resample("M").sum().rolling(12).kurt().reindex(df.index).fillna(0)

    # === 6) TARGET E HISTÓRICO (LISTA B) ===
    events_B = ["1962-10-01", "1966-02-01", "1970-01-01", "1973-10-01", "1987-10-19", "2000-03-01", "2008-09-01", "2020-03-01", "2022-07-01"]
    df["target"] = 0
    for ev in events_B:
        ev_dt = pd.to_datetime(ev)
        df.loc[(df.index < ev_dt) & (df.index >= ev_dt - pd.DateOffset(months=12)), "target"] = 1

    # === 7) XGBOOST (MONOTÓNICO) ===
    features = ["pseudo_vix", "vix_ma6", "vix_std12", "dV", "d2V", "d3V", "d3V_s", "curve", "TED", "M2_yoy", "drawdown_12"]
    X = df[features].ffill().bfill().fillna(0)
    y = df["target"]
    
    # Restricciones monotónicas
    m_const = "(1,1,1,1,1,1,1,-1,1,-1,1)" 
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective":"binary:logistic", "eval_metric":"auc", "max_depth":4, "eta":0.05, "monotone_constraints":m_const}
    bst = xgb.train(params, dtrain, num_boost_round=200)
    
    df["prob_model"] = bst.predict(dtrain)
    return df

# --- EJECUCIÓN ---
try:
    with st.spinner('Procesando datos IGU v5.2...'):
        df_results = run_igu_model()
    
    ultimo = df_results.iloc[-1]
    prob_actual = round(ultimo['prob_model'] * 100, 1)

    # --- DASHBOARD VISUAL ---
    color_riesgo = "#ef4444" if prob_actual > 44 else "#f59e0b" if prob_actual > 20 else "#10b981"
    
    st.markdown(f"""
        <div style="background:#0f172a; color:white; padding:2rem; border-radius:1rem; border-left:10px solid {color_riesgo}">
            <h3 style="margin:0">MONITOR DE RIESGO IGU v5.2 (ML)</h3>
            <h1 style="font-size:4rem; margin:0">{prob_actual}%</h1>
            <p style="font-size:1.2rem">Estado: <b>{"ALERTA CRÍTICA" if prob_actual > 44 else "ESTABLE"}</b></p>
        </div>
    """, unsafe_area_allowed=True)

    # --- GRÁFICO 2022 - PRESENTE ---
    st.subheader("📉 Evolución de Probabilidad (2022 - Actualidad)")
    df_plot = df_results[df_results.index >= "2022-01-01"]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_plot.index, df_plot["prob_model"]*100, color="#6366f1", lw=2.5, label="Probabilidad ML")
    ax.axhline(44, color="red", linestyle="--", label="Umbral Crítico (44%)")
    ax.fill_between(df_plot.index, 44, df_plot["prob_model"]*100, where=(df_plot["prob_model"]*100 >= 44), color='red', alpha=0.3)
    ax.set_ylim(0, 105)
    ax.legend()
    st.pyplot(fig)

    # --- ANÁLISIS DE DERIVADAS ---
    st.subheader("🧬 Análisis de Estructura Interna")
    c1, c2, c3 = st.columns(3)
    c1.metric("Velocidad (dV)", f"{ultimo['dV']:.4f}")
    c2.metric("Aceleración (d2V)", f"{ultimo['d2V']:.4f}")
    c3.metric("Tirón (d3V)", f"{ultimo['d3V']:.4f}")

    st.success(f"**Síntesis Final:** El modelo XGBoost detecta una probabilidad del {prob_actual}%. " 
               f"Las derivadas indican una {'presión al alza' if ultimo['dV'] > 0 else 'relajación'} del riesgo sistémico.")

except Exception as e:
    st.error(f"Error en el motor de cálculo: {e}")
