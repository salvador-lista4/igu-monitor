import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components
import datetime

# --- CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="IGU v5.2 Live Monitor", layout="wide")

# Forzar descarga de datos diaria
@st.cache_data(ttl=86400)
def get_data_and_predict():
    # 1) Descarga S&P500 de forma segura
    sp_data = yf.download("^GSPC", start="1960-01-01", progress=False)
    
    # Esta línea arregla el error: elimina niveles extras si existen
    if isinstance(sp_data.columns, pd.MultiIndex):
        sp_data.columns = sp_data.columns.get_level_values(0)
    
    # Seleccionamos 'Close' o 'Adj Close' según disponibilidad
    col_name = 'Adj Close' if 'Adj Close' in sp_data.columns else 'Close'
    sp = sp_data[[col_name]].rename(columns={col_name: 'price'})
    
    sp["ret"] = np.log(sp["price"]).diff()
    sp["rv30"] = sp["ret"].rolling(30).std() * np.sqrt(252)
    # ... (el resto del código sigue igual)
    
    # 2) Macro de FRED
    fred_codes = {"DGS10": "US10Y", "DGS2": "US2Y", "TEDRATE": "TED"}
    macro_frames = []
    for code, name in fred_codes.items():
        s = pdr.DataReader(code, "fred", start="1960-01-01")
        macro_frames.append(s.rename(columns={code: name}))
    
    macro_df = pd.concat(macro_frames, axis=1).resample("M").last().ffill()
    macro_df["curve"] = macro_df["US10Y"] - macro_df["US2Y"]
    
    # 3) Unión y Derivadas
    df = sp["rv30"].resample("M").last().to_frame("pseudo_vix").join(macro_df).dropna()
    V = -np.log(df["pseudo_vix"].clip(lower=1e-8))
    W, P = 11, 3
    df["V"] = savgol_filter(V, W, P, deriv=0)
    df["dV"] = savgol_filter(V, W, P, deriv=1)
    df["d2V"] = savgol_filter(V, W, P, deriv=2)
    df["d3V"] = savgol_filter(V, W, P, deriv=3)
    
    # 4) Simulación de Predicción (Aquí iría tu modelo XGBoost cargado)
    # Por ahora usamos una función lógica basada en tu modelo para la web
    df["prob_model"] = (1 / (1 + np.exp(-(df["dV"]*10 + df["pseudo_vix"]/50 - 2))))
    return df

df = get_data_and_predict()
ultimo = df.iloc[-1]
prob_actual = round(ultimo['prob_model'] * 100, 1)

# --- INDICADOR VISUAL (HTML/CSS) ---
color_riesgo = "#ef4444" if prob_actual > 44 else "#10b981"
html_card = f"""
<div style="font-family:'Inter',sans-serif; background:#0f172a; color:white; padding:30px; border-radius:15px; border-left: 10px solid {color_riesgo}">
    <div style="display:flex; justify-content:space-between">
        <div>
            <h4 style="margin:0; opacity:0.8; text-transform:uppercase">Riesgo Sistémico IGU v5.2</h4>
            <h1 style="font-size:4rem; margin:10px 0">{prob_actual}%</h1>
            <span style="background:{color_riesgo}; padding:5px 15px; border-radius:20px; font-weight:bold">
                {"ALERTA CRÍTICA" if prob_actual > 44 else "SISTEMA ESTABLE"}
            </span>
        </div>
        <div style="text-align:right">
            <p style="margin:0; opacity:0.7">Pseudo-VIX: <b>{ultimo['pseudo_vix']:.2f}</b></p>
            <p style="margin:0; opacity:0.7">Curva 10Y-2Y: <b>{ultimo['curve']:.2f}</b></p>
            <p style="margin:0; opacity:0.7">Actualizado: {datetime.datetime.now().strftime('%d/%m/%Y')}</p>
        </div>
    </div>
</div>
"""
components.html(html_card, height=220)

# --- GRÁFICO DESDE 2022 ---
st.subheader("📊 Evolución del Riesgo (2022 - Presente)")
df_recent = df[df.index >= "2022-01-01"]

fig, ax = plt.subplots(figsize=(12, 4), facecolor='#f8fafc')
ax.plot(df_recent.index, df_recent['prob_model']*100, color='#6366f1', lw=2, label="Probabilidad IGU")
ax.axhline(44, color='#ef4444', linestyle='--', alpha=0.7, label="Umbral Alerta (44%)")

# Sombreado de alerta
ax.fill_between(df_recent.index, 0, df_recent['prob_model']*100, 
                where=(df_recent['prob_model']*100 >= 44), color='#ef4444', alpha=0.3)

ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.2)
ax.legend()
st.pyplot(fig)

# --- SÍNTESIS DE DERIVADAS ---
st.subheader("🧠 Análisis de Estructura (Derivadas)")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Velocidad (dV)", f"{ultimo['dV']:.4f}", help="Cambio inmediato en la estabilidad")
    st.caption("Indica si el riesgo está ganando tracción en este momento.")

with col2:
    st.metric("Aceleración (d2V)", f"{ultimo['d2V']:.4f}")
    st.caption("Indica si la presión sobre el sistema se está intensificando.")

with col3:
    st.metric("Tirón (d3V)", f"{ultimo['d3V']:.4f}")
    st.caption("Cambio en la aceleración: clave para detectar puntos de quiebre.")

# --- SÍNTESIS FINAL ---
st.info(f"""
### 📝 Síntesis Técnica
El sistema presenta una probabilidad del **{prob_actual}%**. 
* **Dinámica:** Con un dV de **{ultimo['dV']:.4f}**, la velocidad de cambio es {'creciente' if ultimo['dV'] > 0 else 'decreciente'}.
* **Conclusión:** {'Se recomienda precaución extrema ya que el indicador supera el umbral crítico del 44%.' if prob_actual > 44 else 'Los parámetros se mantienen dentro de los rangos de control histórico.'}
""")

