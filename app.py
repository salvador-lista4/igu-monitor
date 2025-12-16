# IGU v5.2 — Versión B (crisis + mini-crisis) — corregido y robusto
# - Pseudo-VIX (homogéneo 1960–2025)
# - Outer join + imputación
# - Target: lista B (crisis grandes + mini-crisis) dentro de 12 meses
# - XGBoost monotónico + balanceo
# - Salidas: CSVs, modelo, métricas, SHAP (si disponible)
# ===================================================================

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

# === 0) PARAMETERS / HYPER =========================
START = "1959-01-01"
END   = "2025-12-31"
SG_WINDOW = 11   # odd, smaller than before
SG_POLY   = 3
TARGET_HORIZON_MONTHS = 12
TRAIN_END = "1999-12-31"
VAL_END   = "2010-12-31"
# ===================================================

print("1) Descargando S&P500 y construyendo pseudo-VIX...")
sp = yf.download("^GSPC", start=START, end=END, progress=False, auto_adjust=False)
if isinstance(sp.columns, pd.MultiIndex):
    sp.columns = sp.columns.get_level_values(0)
price_col = "Adj Close" if "Adj Close" in sp.columns else "Close"
sp = sp[[price_col]].rename(columns={price_col: "price"})
sp["ret"] = np.log(sp["price"]).diff()
sp = sp.dropna()

# realized vol 30-day annualized (pseudo-VIX)
sp["rv30"] = sp["ret"].rolling(30, min_periods=10).std() * np.sqrt(252)
pseudo_monthly = sp["rv30"].resample("M").last().to_frame("pseudo_vix")
print(" pseudo-VIX monthly points:", len(pseudo_monthly))

# === 2) Download macro series from FRED (outer join) =========
print("2) Descargando series FRED (outer join)...")
fred_codes = {
    "DGS10": "US10Y",
    "DGS2":  "US2Y",
    "BAMLH0A0HYM2": "HY_spread",   # may be missing in some environments
    "TEDRATE": "TED",
    "M2SL": "M2"
}

macro_frames = []
for code, name in fred_codes.items():
    try:
        s = pdr.DataReader(code, "fred", start="1960-01-01", end=END)
        s = s.rename(columns={code: name})
        # monthly
        s = s.resample("M").last()
        macro_frames.append(s)
        print("  OK:", code)
    except Exception as e:
        print("  WARNING: no pude descargar", code, "→ se omitirá si es crítico.", str(e))

if len(macro_frames) == 0:
    raise RuntimeError("No se pudo descargar ninguna serie de FRED. Revisa conexión o códigos.")

macro_df = pd.concat(macro_frames, axis=1)
# compute curve
if ("US10Y" in macro_df.columns) and ("US2Y" in macro_df.columns):
    macro_df["curve"] = macro_df["US10Y"] - macro_df["US2Y"]
else:
    macro_df["curve"] = np.nan
print(" macro monthly points:", len(macro_df))

# === 3) Merge everything with outer join and robust imputation ===
print("3) Unión outer PV + macro y limpieza/imputación...")
df = pseudo_monthly.join(macro_df, how="outer")
# Keep window 1960-01 onward
df = df[(df.index >= "1960-01-01") & (df.index <= END)].sort_index()

# Impute: linear interpolation where sensible, then forward/back fill
df["pseudo_vix"] = df["pseudo_vix"].interpolate(limit=6).ffill().bfill()
# For macro columns: small gaps interpolate, big gaps ffill
for col in df.columns:
    if col == "pseudo_vix": continue
    df[col] = df[col].interpolate(limit=3).ffill().bfill()

# Drop rows still with NA in key series (if any)
df = df.dropna(subset=["pseudo_vix"])
print("Rows after imputation:", len(df))

# === 4) Construct IGU potential and derivatives with stable SG params ===
print("4) Calculando V = -ln(pseudo_vix) y derivadas (SG)...")
vol = df["pseudo_vix"].values
V = -np.log(np.clip(vol, 1e-8, None))   # avoid zeros
# pick window adaptively
W = SG_WINDOW
if W >= len(V): W = (len(V)//2)*2 - 1
if W < 3: W = 3
if W % 2 == 0: W += 1
P = SG_POLY if SG_POLY < W else max(1, W-2)

V_s = savgol_filter(V, W, P, deriv=0, mode="interp")
dV  = savgol_filter(V, W, P, deriv=1, delta=1.0, mode="interp")
d2V = savgol_filter(V, W, P, deriv=2, delta=1.0, mode="interp")
d3V = savgol_filter(V, W, P, deriv=3, delta=1.0, mode="interp")

df["V"] = V_s
df["dV"] = dV
df["d2V"] = d2V
df["d3V"] = d3V

# small smoothing of d3V (EMA)
df["d3V_s"] = pd.Series(df["d3V"]).ewm(span=4).mean().values

# === 5) Feature engineering ===
print("5) Feature engineering (rolling stats, drawdowns, M2 growth)...")
# rolling stats of pseudo_vix
df["vix_ma6"] = df["pseudo_vix"].rolling(6).mean()
df["vix_std12"] = df["pseudo_vix"].rolling(12).std()

# M2 yoy (if present)
if "M2" in df.columns:
    df["M2_yoy"] = df["M2"].pct_change(12)
else:
    df["M2_yoy"] = 0.0

# Price-derived features (align monthly prices)
monthly_price = sp["price"].resample("M").last().reindex(df.index)
df["price"] = monthly_price.values
df["roll_max_12"] = df["price"].rolling(12, min_periods=1).max()
df["drawdown_12"] = df["price"] / df["roll_max_12"] - 1.0

# kurtosis of monthly returns (proxy)
monthly_ret = sp["ret"].resample("M").sum().reindex(df.index)
df["ret_kurt_12"] = monthly_ret.rolling(12).kurt().fillna(0.0)

# fill any remaining NA with small neutral values
df = df.fillna(0.0)

# === 6) Build target using LIST B events (crisis+mini) =========
print("6) Construyendo target a partir de lista B de eventos (horizonte 12m)...")
# definitive list B (approx dates) - you can refine dates later
events_B = {
    "1962_flash":      pd.Timestamp("1962-10-01"),
    "1966_credit":     pd.Timestamp("1966-02-01"),
    "1970_recession":  pd.Timestamp("1970-01-01"),
    "1973_oil":        pd.Timestamp("1973-10-01"),
    "1987_black_monday": pd.Timestamp("1987-10-19"),
    "1994_bond":       pd.Timestamp("1994-10-01"),
    "1997_asia":       pd.Timestamp("1997-07-01"),
    "1998_ltcM":       pd.Timestamp("1998-09-01"),
    "2000_dotcom":     pd.Timestamp("2000-03-01"),
    "2008_subprime":   pd.Timestamp("2008-09-01"),
    "2011_euro":       pd.Timestamp("2011-08-01"),
    "2015_china":      pd.Timestamp("2015-08-01"),
    "2018_q4":         pd.Timestamp("2018-10-01"),
    "2022_tightening": pd.Timestamp("2022-07-01"),
    "2020_covid":      pd.Timestamp("2020-03-01")
}

# create binary target: 1 if any event in events_B occurs within next TARGET_HORIZON_MONTHS
dates = df.index.to_list()
target = np.zeros(len(df), dtype=int)
for i, dt in enumerate(dates):
    for ev_name, ev_date in events_B.items():
        # consider only future events within horizon
        delta_months = (ev_date.year - dt.year) * 12 + (ev_date.month - dt.month)
        if 0 < delta_months <= TARGET_HORIZON_MONTHS:
            target[i] = 1
            break
df["target_12m_B"] = target
print("Target positives total:", int(df["target_12m_B"].sum()))

# === 7) Feature matrix and scaling ============================
features = [
    "pseudo_vix", "vix_ma6", "vix_std12",
    "dV", "d2V", "d3V", "d3V_s",
    "curve", "HY_spread" if "HY_spread" in df.columns else "US10Y",
    "TED", "M2_yoy",
    "drawdown_12", "ret_kurt_12"
]
# keep only existing features
features = [f for f in features if f in df.columns]
X = df[features].copy()
y = df["target_12m_B"].copy()

# forward/backward fill small holes then scale
X = X.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
scaler = StandardScaler()
Xs = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

# === 8) train/val/test split (time-based) =======================
train_idx = Xs.index <= TRAIN_END
val_idx   = (Xs.index > TRAIN_END) & (Xs.index <= VAL_END)
test_idx  = Xs.index > VAL_END

X_train, y_train = Xs[train_idx], y[train_idx]
X_val, y_val = Xs[val_idx], y[val_idx]
X_test, y_test = Xs[test_idx], y[test_idx]

print("Splits sizes: train", X_train.shape, "val", X_val.shape, "test", X_test.shape)
print("Target counts (train/val/test):", int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))

# If training positives are too few, extend training window backward (safety)
if y_train.sum() < 5:
    print("Advertencia: pocos positivos en train. Extendiendo train hasta 2009 para más ejemplos.")
    TRAIN_END = "2009-12-31"
    train_idx = Xs.index <= TRAIN_END
    val_idx = (Xs.index > TRAIN_END) & (Xs.index <= VAL_END)
    X_train, y_train = Xs[train_idx], y[train_idx]
    X_val, y_val = Xs[val_idx], y[val_idx]
    print("Nuevos splits: train", X_train.shape, "val", X_val.shape)
    print("Target counts (train/val):", int(y_train.sum()), int(y_val.sum()))

# === 9) monotonic constraints vector (match Xs.columns order) =====
# define expected direction: +1 means feature up -> risk up, -1 opposite, 0 no constraint
monotone_map = {}
for f in Xs.columns:
    if f in ("pseudo_vix","vix_ma6","vix_std12","dV","d2V","d3V","d3V_s","HY_spread","TED","ret_kurt_12","drawdown_12"):
        monotone_map[f] = 1
    elif f in ("curve","M2_yoy"):
        monotone_map[f] = -1
    else:
        monotone_map[f] = 0
monotone_constraints = [monotone_map.get(f, 0) for f in Xs.columns]
print("Monotonic constraints:", monotone_constraints)

# === 10) XGBoost training with scale_pos_weight and monotone constraints ====
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

neg = (y_train==0).sum()
pos = (y_train==1).sum() if (y_train==1).sum()>0 else 1
scale_pos_weight = neg/pos
params = {
    "objective":"binary:logistic",
    "eval_metric":"auc",
    "tree_method":"hist",
    "max_depth":4,
    "eta":0.05,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "monotone_constraints":"(" + ",".join([str(int(x)) for x in monotone_constraints]) + ")",
    "scale_pos_weight": float(scale_pos_weight)
}
print("XGB params:", params)
evallist = [(dtrain, "train"), (dval, "val")]
bst = xgb.train(params, dtrain, num_boost_round=1000, evals=evallist, early_stopping_rounds=50, verbose_eval=50)

# === 11) Evaluate ===============================================
proba_val = bst.predict(dval)
proba_test = bst.predict(dtest)
auc_val = roc_auc_score(y_val, proba_val) if y_val.sum()>0 else np.nan
auc_test = roc_auc_score(y_test, proba_test) if y_test.sum()>0 else np.nan
print("ROC AUC val:", auc_val, " test:", auc_test)

prec, rec, thr = precision_recall_curve(y_test, proba_test)
pr_auc = auc(rec, prec)
print("PR AUC test:", pr_auc)

# choose threshold using Youden on val
fpr, tpr, tthr = roc_curve(y_val, proba_val)
if len(tthr)>0:
    youden = np.argmax(tpr - fpr)
    thr_best = tthr[youden]
else:
    thr_best = 0.5
print("Threshold (Youden on val):", thr_best)

preds_test = (proba_test >= thr_best).astype(int)
acc_test = accuracy_score(y_test, preds_test)
print("Test accuracy at threshold:", acc_test)

# === 12) Save outputs ============================================
print("Guardando resultados y CSVs...")
df_out = df.loc[Xs.index].copy()
df_out = df_out.assign(prob_model = pd.Series(bst.predict(xgb.DMatrix(Xs)), index=Xs.index),
                       pred_model = lambda d: (d["prob_model"] >= thr_best).astype(int),
                       target = y)
df_out.to_csv("IGU_v5_2_full_dataset_with_preds.csv")
bst.save_model("IGU_v5_2_xgb.model")
print("Archivos guardados.")

# === 13) Predicted events in test period (post val) = actionable =====
pred_events = df_out[(df_out.index > VAL_END) & (df_out["pred_model"]==1)]
print("Predicted events (post-val):")
print(pred_events[["prob_model","pred_model","target"]])

# === 14) SHAP explainability (optional, may be heavy) ============
try:
    import shap
    explainer = shap.TreeExplainer(bst)
    shap_vals = explainer.shap_values(xgb.DMatrix(Xs))
    shap_summary = pd.DataFrame(np.abs(shap_vals).mean(0), index=Xs.columns, columns=["mean_abs_shap"]).sort_values("mean_abs_shap", ascending=False)
    print("SHAP top features:")
    print(shap_summary.head(12))
    shap.summary_plot(shap_vals, Xs, show=False)
    plt.savefig("IGU_v5_2_shap_summary.png", dpi=200)
except Exception as e:
    print("SHAP no disponible o falló:", e)

# === 15) Quick diagnostics plots =================================
plt.figure(figsize=(12,5))
plt.plot(df_out.index, df_out["prob_model"], label="prob_model")
plt.plot(df_out.index, df_out["target"], label="target (events)", alpha=0.6)
plt.axvline(pd.to_datetime(TRAIN_END), color="k", linestyle="--", alpha=0.5)
plt.axvline(pd.to_datetime(VAL_END), color="k", linestyle=":", alpha=0.5)
plt.legend()
plt.title("IGU v5.2 — Probabilidades y eventos (1960–2025)")
plt.tight_layout()
plt.savefig("IGU_v5_2_prob_vs_target.png", dpi=200)
plt.show()

print("LISTO: revisá 'IGU_v5_2_full_dataset_with_preds.csv' y 'IGU_v5_2_xgb.model'.")

# ... (Todo tu código actual de procesamiento va aquí arriba) ...

import streamlit as st
import streamlit.components.v1 as components

# Al final de tu script, extraemos los últimos valores calculados
ultimo_dato = df_out.iloc[-1]
prob_actual = round(ultimo_dato['prob_model'] * 100, 1)
vix_actual = round(ultimo_dato['pseudo_vix'], 2)
fecha_actual = ultimo_dato.name.strftime("%b %d, %Y")

# Definimos el estado basado en la probabilidad
estado = "ALTO" if prob_actual > 60 else "MEDIO" if prob_actual > 30 else "BAJO"
color = "#ef4444" if estado == "ALTO" else "#f59e0b" if estado == "MEDIO" else "#10b981"

# Inyectamos el HTML que te pasé antes, pero con los datos de tu Python
html_final = f"""
<div style="font-family: 'Inter', sans-serif;">
    <div style="background: #0f172a; color: white; padding: 20px; border-radius: 12px 12px 0 0;">
        <h2>📈 IGU v5.2 Live</h2>
        <p>Actualizado: {fecha_actual}</p>
    </div>
    <div style="padding: 30px; border: 1px solid #e2e8f0; background: white;">
        <span style="color: #64748b; font-weight: bold;">PROBABILIDAD DE CRISIS (12M)</span>
        <div style="display: flex; align-items: baseline; gap: 20px;">
            <h1 style="font-size: 4rem; color: {color}; margin: 10px 0;">{estado}</h1>
            <span style="font-size: 2rem; color: #64748b;">{prob_actual}%</span>
        </div>
        <div style="background: #f1f5f9; height: 10px; border-radius: 5px;">
            <div style="width: {prob_actual}%; background: {color}; height: 100%; border-radius: 5px;"></div>
        </div>
    </div>
</div>
"""

# Esto hace que Streamlit muestre tu diseño en la web
st.set_page_config(page_title="IGU v5.2 Monitor", layout="wide")
components.html(html_final, height=400)

# También puedes mostrar los gráficos que genera tu código automáticamente
st.pyplot(plt)
