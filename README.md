# Sistema Avanzado de Pronóstico de Demanda

Proyecto de predicción de demanda utilizando series temporales con modelos LSTM en TensorFlow. Incluye generación de datos sintéticos, entrenamiento con validación cruzada, y aplicación web interactiva con Streamlit para visualizar pronósticos y evaluación del modelo.

---

## Estructura del Proyecto

```text
PROJECT_V2/
├── data/
│   ├── processed/
│   │   └── sales_data.csv          # Datos de ventas procesados
│   └── raw/                        # Datos crudos (fuente)
├── logs/                           # Registros de entrenamiento
├── mlruns/                         # Artefactos de MLflow
├── models/
│   ├── v1/
│   │   ├── final_model.keras       # Modelo entrenado final
│   │   ├── fold_0.keras
│   │   ├── fold_1.keras
│   │   ├── fold_2.keras
│   │   ├── fold_3.keras
│   │   └── fold_4.keras
│   └── scaler.pkl                  # Scaler para estandarización
├── utils/
│   ├── data_processing.py          # Generación y preparación de datos
│   └── model_evaluation.py         # Evaluación y visualización
├── app.py                          # Aplicación Streamlit
├── train.py                        # Script de entrenamiento
├── config.yaml                     # Configuraciones generales
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Este archivo


## Requisitos

- Python 3.10 o superior  
- GPU opcional, pero recomendado para acelerar entrenamiento  
- Recomendado crear un entorno virtual (venv o conda)

---

## Instalación

### 1. Clonar el repositorio:

git clone https://github.com/MrAleexx/RNN_Course_IA

### 2. Crear y activar entorno virtual

python -m venv venv

#### Windows

venv\Scripts\activate

#### Linux/macOS

source venv/bin/activate

### 3. Instalar dependencias:

pip install -r requirements.txt

## Uso

### 2. Entrenar modelo

Ejecutar el script de entrenamiento que realiza validación cruzada y guarda modelos:

python train.py

### 3. Ejecutar aplicación Streamlit

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

streamlit run app.py
