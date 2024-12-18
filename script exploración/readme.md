# Predicción de Concentración de Antígenos en Producción de Vacunas

## 🎯 Descripción del Proyecto
Este proyecto, desarrollado para el UniversityHack 2024, se centra en la optimización de la producción de antígenos para el desarrollo de vacunas. Implementamos un ensemble de modelos de machine learning para predecir la concentración final del antígeno, considerando múltiples variables del proceso productivo.

## 📁 Estructura del Repositorio

```
├── script exploración/
│   ├── data/                    # Datos iniciales
│   ├── processed_data/          # Datos procesados
│   ├── data_cleaning_and_analysis.ipynb  # Notebook de preprocesamiento
│   └── data_cleaning_and_analysis.py     # Script de preprocesamiento
│
└── script predicción/
    ├── tmp/                     # Modelos optimizados
    ├── log/                     # Registros de entrenamiento
    ├── dataloader.py           # Carga y escalado de datos
    ├── model.py               # Definición de modelos
    ├── pipeline.py            # Pipeline de entrenamiento
    └── main.py               # Script principal
```

## 📊 Rendimiento de los Modelos

| Modelo | RMSE | Desviación Estándar | Emisiones CO2 (g) | Tiempo (s) |
|--------|------|---------------------|-------------------|------------|
| Ensemble | 249.53 | ±23.47 | 0.00203 | 5.58 |
| CatBoost | 252.97 | ±35.58 | 0.00090 | 2.58 |
| Random Forest | 257.83 | ±26.23 | 0.00057 | 1.51 |
| XGBoost | 261.74 | ±23.70 | 0.00152 | 4.30 |
| LightGBM | 264.67 | ±22.55 | 0.00079 | 2.11 |
| Gradient Boosting | 263.52 | ±17.92 | 0.00056 | 1.68 |

## 🚀 Características Principales

- **Ensemble Avanzado**: Implementación de voting con pesos optimizados
- **Explicabilidad**: Análisis detallado mediante SHAP
- **Sostenibilidad**: Monitorización de emisiones CO2 con CodeCarbon
- **Validación Robusta**: Sistema de validación cruzada 5-fold
- **Preprocesamiento**: Manejo específico de valores nulos y escalado de datos

## 🛠️ Requisitos

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost shap codecarbon
```

## 📈 Resultados Destacados

- 🏆 El ensemble superó consistentemente a los modelos individuales
- 📊 Distribución de pesos optimizada (CatBoost: 6.0, RandomForest: 3.5)
- 🌱 Emisiones totales inferiores a 0.003g CO2
- 🔍 Identificación de variables clave mediante análisis SHAP

## 🤝 Contribuciones
Desarrollado por el equipo HAL9000 para UniversityHack 2024.

## 📝 Licencia
Este proyecto está bajo la licencia MIT.