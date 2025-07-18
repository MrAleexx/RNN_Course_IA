#model_evaluation.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ───────────────────────────────────────────
# 1. Métricas sobre un set (test o val)
# ───────────────────────────────────────────
def evaluate_model(model, X, y, scaler):
    """Calcula MAE, RMSE, MAPE y R² en datos escalados."""
    y_pred = model.predict(X, verbose=0).flatten()
    y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_real = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    return {
        "mae": mean_absolute_error(y_real, y_pred_real),
        "rmse": np.sqrt(mean_squared_error(y_real, y_pred_real)),
        "mape": np.mean(np.abs((y_real - y_pred_real) / y_real)) * 100,
        "r2": r2_score(y_real, y_pred_real),
    }


# ───────────────────────────────────────────
# 2. Gráfico real vs pred
# ───────────────────────────────────────────
def plot_predictions(y_true, y_pred, dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, name="Real", line=dict(color="blue")))
    fig.add_trace(
        go.Scatter(x=dates, y=y_pred, name="Predicción", line=dict(color="orange", dash="dot"))
    )
    fig.update_layout(
        title="Real vs Predicción",
        xaxis_title="Fecha",
        yaxis_title="Demanda",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ───────────────────────────────────────────
# 3. Forecast Monte-Carlo con bandas de confianza
# ───────────────────────────────────────────
def generate_forecast(
    model,
    scaler,
    last_window: np.ndarray,
    last_date: pd.Timestamp,
    n_days: int,
    n_samples: int = 1000,
    ci: int = 80,
):
    seq_len = len(last_window)
    win_scaled = scaler.transform(last_window.reshape(-1, 1)).flatten()

    paths = []
    for _ in range(n_samples):
        w = win_scaled.copy()
        sim = []
        for _ in range(n_days):
            y_hat = model.predict(w.reshape(1, seq_len, 1), verbose=0)[0, 0]
            sim.append(y_hat)
            w = np.append(w[1:], y_hat)
        paths.append(scaler.inverse_transform(np.array(sim).reshape(-1, 1)).flatten())

    paths = np.array(paths)  # shape (n_samples, n_days)

    mean_fc = paths.mean(axis=0)
    lower = np.percentile(paths, (100 - ci) / 2, axis=0)
    upper = np.percentile(paths, 100 - (100 - ci) / 2, axis=0)

    f_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    return {"dates": f_dates, "mean": mean_fc, "lower": lower, "upper": upper}


# ───────────────────────────────────────────
# 4. Backtesting deslizante
# ───────────────────────────────────────────
def backtest_model(model, scaler, series, seq_len: int, horizon: int = 7):
    X, y = [], []
    for i in range(seq_len, len(series) - horizon):
        X.append(series[i - seq_len : i])
        y.append(series[i : i + horizon])

    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)
    X_scaled = scaler.transform(X.reshape(-1, 1)).reshape(-1, seq_len, 1)

    preds = []
    for window in X_scaled:
        w = window.copy()
        pred = []
        for _ in range(horizon):
            y_hat = model.predict(w.reshape(1, seq_len, 1), verbose=0)[0, 0]
            pred.append(y_hat)
            w = np.append(w[1:], y_hat).reshape(seq_len, 1)
        preds.append(scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten())

    preds = np.array(preds)

    return {
        "mae": np.mean([mean_absolute_error(y[i], preds[i]) for i in range(len(y))]),
        "rmse": np.mean([np.sqrt(mean_squared_error(y[i], preds[i])) for i in range(len(y))]),
        "mape": np.mean([np.mean(np.abs((y[i] - preds[i]) / y[i])) * 100 for i in range(len(y))]),
        "r2": np.mean([r2_score(y[i], preds[i]) for i in range(len(y))]),
        "y_true": y[-1],
        "y_pred": preds[-1],
        "dates": pd.date_range(periods=horizon, end=pd.Timestamp.today()),
    }

