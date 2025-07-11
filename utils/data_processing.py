#data_processing.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ───────────────────────────────────────────
# 1. Generación de datos sintéticos
# ───────────────────────────────────────────
def create_synthetic_data(cfg: dict) -> pd.DataFrame:
    """Genera una tabla de ventas diarias con tendencia, estacionalidades y promos."""
    dates = pd.date_range(start=cfg["start_date"], periods=cfg["n_days"], freq="D")

    trend = np.linspace(100, 300, len(dates))
    weekly = 50 * np.sin(2 * np.pi * dates.dayofweek / 7)
    yearly = 30 * np.sin(2 * np.pi * (dates.dayofyear - 1) / 365)

    promos = np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])
    promo_effect = promos * np.random.normal(50, 10, len(dates))

    df = pd.DataFrame(
        {
            "date": dates,
            "store_id": np.random.choice(cfg["store_ids"], len(dates)),
            "product_id": np.random.choice(cfg["product_ids"], len(dates)),
            "sales": trend + weekly + yearly + promo_effect + np.random.poisson(50, len(dates)),
            "price": np.random.uniform(5, 20, len(dates)),
            "promotion": promos,
            "is_weekend": (dates.dayofweek >= 5).astype(int),
            "month": dates.month,
        }
    )
    return df


# ───────────────────────────────────────────
# 2. Carga (o genera) los datos
# ───────────────────────────────────────────
def load_data(data_path: str | Path, synthetic_cfg: dict) -> pd.DataFrame:
    """
    Lee el CSV indicado.  
    Si no existe, genera datos sintéticos y los guarda.
    """
    data_path = Path(data_path)
    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=["date"])
    else:
        df = create_synthetic_data(synthetic_cfg)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"⚠️  Datos sintéticos generados y guardados en {data_path}")
    return df


# ───────────────────────────────────────────
# 3. Ventanas para entrenamiento LSTM
# ───────────────────────────────────────────
def prepare_time_series(
    df: pd.DataFrame, target_col: str, test_size: float = 0.1, seq_len: int = 60
):
    """Convierte la serie univariada en (X, y) escalados más scaler."""
    scaler = StandardScaler()
    series = df[target_col].values.reshape(-1, 1)
    scaled = scaler.fit_transform(series)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len : i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)

    split = int(len(X) * (1 - test_size))
    return {"X_train": X[:split], "y_train": y[:split], "X_test": X[split:], "y_test": y[split:]}, scaler


# ───────────────────────────────────────────
# 4. Última ventana para inferencia
# ───────────────────────────────────────────
def prepare_forecast_data(df: pd.DataFrame, target_col: str, seq_len: int):
    """Devuelve la última ventana y la fecha final para arrancar el forecast."""
    last_window = df[target_col].iloc[-seq_len:].values.astype(float)
    last_date = pd.to_datetime(df["date"].iloc[-1])
    return last_window, last_date
