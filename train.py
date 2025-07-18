# train.py

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf
import pickle
import mlflow
from datetime import datetime
from utils.data_processing import create_synthetic_data, prepare_time_series
from utils.model_evaluation import evaluate_model

# Configuración
with open('config.yaml') as f:
    config = yaml.safe_load(f)

SEED = config['random_seed']
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Inicializar MLflow
mlflow.set_tracking_uri(config['mlflow_uri'])
mlflow.set_experiment(config['experiment_name'])


def build_model(input_shape):
    """ Modelo LSTM """
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True,
                        kernel_regularizer=l2(0.01)),
                        input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=config['learning_rate'],
                    clipnorm=config['clipnorm'])
    model.compile(optimizer=optimizer,
                    loss='mae',
                    metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()])
    return model


def main():
    # Cargar o generar datos
    data_path = Path(config['data_path'])
    if not data_path.exists():
        df = create_synthetic_data(config['synthetic_data'])
        data_path.parent.mkdir(exist_ok=True)
        df.to_csv(data_path, index=False)
        print("⚠️ Datos sintéticos generados")
    else:
        df = pd.read_csv(data_path, parse_dates=['date'])

    # Preparar serie temporal
    series, scaler = prepare_time_series(df,
                                        target_col=config['target_column'],
                                        test_size=config['test_size'],
                                        seq_len=config['sequence_length'])

    # Cross-validation temporal
    tscv = TimeSeriesSplit(n_splits=config['n_splits'])

    with mlflow.start_run():
        # Log de parámetros
        mlflow.log_params(config)

        # Entrenamiento con CV
        for fold, (train_idx, val_idx) in enumerate(tscv.split(series['X_train'])):
            print(f"\nFold {fold + 1}")
            X_train, X_val = series['X_train'][train_idx], series['X_train'][val_idx]
            y_train, y_val = series['y_train'][train_idx], series['y_train'][val_idx]

            model = build_model((config['sequence_length'], 1))

            callbacks = [
                EarlyStopping(patience=config['patience'],
                                restore_best_weights=True,
                                monitor='val_loss'),
                ModelCheckpoint(f"models/v1/fold_{fold}.keras",
                                save_best_only=True),
                TensorBoard(log_dir=f"logs/fold_{fold}")
            ]

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                callbacks=callbacks,
                verbose=1
            )

            # Evaluación y logging
            metrics = evaluate_model(model, X_val, y_val, scaler)
            for metric, value in metrics.items():
                mlflow.log_metric(f"val_{metric}", value)

        # Entrenamiento final con todos los datos
        final_model = build_model((config['sequence_length'], 1))
        final_model.fit(
            series['X_train'], series['y_train'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            verbose=1
        )

        # Guardar artefactos
        model_dir = Path("models") / "v1"
        model_dir.mkdir(exist_ok=True)

        final_model.save(model_dir / "final_model.keras")
        with open(model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        mlflow.log_artifacts(model_dir)
        mlflow.sklearn.log_model(scaler, "scaler")

        print("✅ Entrenamiento completado y artefactos guardados")


if __name__ == "__main__":
    main()