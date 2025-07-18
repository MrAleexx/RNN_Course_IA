# app.py
import os
from datetime import datetime
import streamlit as st
import yaml
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from pathlib import Path
import mlflow
import pickle

from utils.data_processing import load_data, prepare_forecast_data
from utils.model_evaluation import (
    generate_forecast,
    backtest_model,
    plot_predictions,
)

# ───────────────────────────────────────────
# Configuración global
# ───────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="Sistema de Pronóstico de Demanda",
                    page_icon="📊", layout="wide")


# ───────────────────────────────────────────
# Carga de modelo y scaler
# ───────────────────────────────────────────
@st.experimental_singleton
def load_model_artifacts():
    try:
        model = mlflow.keras.load_model(
            f"models:/{config['model_name']}/Production")
        scaler = mlflow.sklearn.load_model(
            f"models:/{config['scaler_name']}/Production")
    except Exception:
        model_path = Path("models/v1/final_model.keras")
        scaler_path = Path("models/v1/scaler.pkl")
        if not model_path.exists() or not scaler_path.exists():
            st.error("Modelo no encontrado. Ejecuta primero train.py.")
            st.stop()
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    return model, scaler


# ───────────────────────────────────────────
# App principal
# ───────────────────────────────────────────
def main():
    st.title("📊 Sistema Pronóstico de Demanda")

    # Datos y modelo
    df = load_data(config["data_path"], config["synthetic_data"])
    model, scaler = load_model_artifacts()

    # — Sidebar —
    with st.sidebar:
        st.header("🔧 Parámetros")
        store_id = st.selectbox("Tienda", sorted(df["store_id"].unique()))
        product_id = st.selectbox(
            "Producto", sorted(df["product_id"].unique()))
        end_date = st.date_input(
            "Fecha final histórica", df["date"].max(), min_value=df["date"].min(), max_value=df["date"].max()
        )
        n_days = st.slider("Horizonte de pronóstico (días)", 1, 30, 7)
        ci = st.slider("Intervalo de confianza (%)", 50, 95, 80)

    # Filtrado de histórico
    df_hist = df[
        (df["store_id"] == store_id)
        & (df["product_id"] == product_id)
        & (df["date"] <= pd.to_datetime(end_date))
    ]

    if len(df_hist) < config["sequence_length"]:
        st.error(
            f"Se necesitan al menos {config['sequence_length']} días de historial.")
        st.stop()

    # Preparar ventana y forecast
    last_window, last_date = prepare_forecast_data(
        df_hist, target_col=config["target_column"], seq_len=config["sequence_length"]
    )
    st.write("Ventana escalada OK: ",last_window.shape," ", "Última fecha:", last_date)
    
    try:
        forecast = generate_forecast(
            model,
            scaler,
            last_window,
            last_date,
            n_days=n_days,
            n_samples=500,   
            ci=ci,
        )

        # En la función main(), modifica la sección del botón:
        if st.button("💾 Guardar pronóstico"):
            output_path = save_forecast(forecast, df_hist, config["target_column"])
            st.success(f"Pronóstico guardado en: {output_path}")
            
            # Opción para descargar directamente
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=f,
                    file_name=f"pronostico_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.exception(e)     # muestra la traza completa
        st.stop()

    # — Visualización —
    st.subheader("📈 Pronóstico")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_hist["date"],
            y=df_hist[config["target_column"]],
            name="Histórico",
            line=dict(color="#1f77b4"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["dates"],
            y=forecast["mean"],
            name="Pronóstico",
            line=dict(color="#ff7f0e", dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(forecast["dates"]) + list(forecast["dates"])[::-1],
            y=list(forecast["upper"]) + list(forecast["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"Intervalo {ci}%",
        )
    )

    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Demanda",
        hovermode="x unified",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # — Backtest —
    st.subheader("📊 Evaluación del Modelo")
    if len(df_hist) > n_days:
        with st.expander("Evaluación histórica"):
            metrics = backtest_model(
                model,
                scaler,
                df_hist[config["target_column"]].values,
                seq_len=config["sequence_length"],
                horizon=n_days,
            )

            cols = st.columns(4)
            cols[0].metric("MAE", f"{metrics['mae']:.2f}")
            cols[1].metric("RMSE", f"{metrics['rmse']:.2f}")
            cols[2].metric("MAPE", f"{metrics['mape']:.2f}%")
            cols[3].metric("R²", f"{metrics['r2']:.2f}")

            st.plotly_chart(
                plot_predictions(metrics["y_true"],
                                metrics["y_pred"], metrics["dates"]),
                use_container_width=True,
            )

def save_forecast(forecast: dict, historical_data: pd.DataFrame, target_col: str) -> str:
    """
    Guarda el pronóstico en una carpeta con timestamp.
    Devuelve la ruta al archivo guardado.
    """
    forecasts_dir = Path("forecasts")
    forecasts_dir.mkdir(exist_ok=True)
    
    # Nombre de carpeta con timestamp válido para Windows
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    forecast_dir = forecasts_dir / f"{timestamp}_forecast"
    forecast_dir.mkdir()
    
    # Generar el DataFrame (igual que antes)
    forecast_df = pd.DataFrame({
        "Fecha": forecast["dates"],
        "Pronóstico": forecast["mean"],
        "Límite_Inferior": forecast["lower"],
        "Límite_Superior": forecast["upper"],
        "Tipo": "Pronóstico"
    })
    
    historical_df = historical_data[["date", target_col]].copy()
    historical_df.columns = ["Fecha", "Pronóstico"]
    historical_df["Límite_Inferior"] = None
    historical_df["Límite_Superior"] = None
    historical_df["Tipo"] = "Histórico"
    
    full_df = pd.concat([historical_df, forecast_df])
    
    # Guardar en la carpeta con timestamp
    timestamped_path = forecast_dir / "forecast.csv"
    full_df.to_csv(timestamped_path, index=False)
    
    # Guardar también como "latest" (alternativa para Windows)
    latest_path = forecasts_dir / "latest_forecast.csv"
    full_df.to_csv(latest_path, index=False)
    
    return str(timestamped_path)


if __name__ == "__main__":
    main()
