# Modelo de Predicción de Churn — Clientes Bancarios

## Contexto

Prueba técnica para el rol de Analista Jr. de Ciencia de Datos en una compañía financiera que ofrece ahorro, inversión, pensiones y seguros. El área comercial necesita anticipar clientes con mayor probabilidad de fuga para priorizar acciones de retención.

## Dataset

**Bank Customer Churn Modelling** (Kaggle) — 10,000 registros de clientes de un banco europeo con variables demográficas, financieras y de comportamiento.

- **Variable objetivo:** `Exited` (1 = cliente abandonó, 0 = se quedó)
- **Tasa de churn:** ~20.4% (clasificación desbalanceada)

## Estructura del Notebook

| Sección | Descripción |
|---------|-------------|
| 1. Entendimiento del problema | Formulación del objetivo analítico y pregunta de negocio |
| 2. Lectura, limpieza y preparación | Inspección de nulos, duplicados, outliers (IQR), tipos de datos y codificación (One-Hot Encoding) |
| 3. Datos semiestructurados (JSON) | Archivo `business_context.json` con reglas de negocio: segmentos de edad, costos de adquisición/retención, niveles de vinculación y segmento de vida financiera |
| 4. Feature Engineering | Creación de variables: `BalanceSalaryRatio`, `TenureAgeRatio`, `HasBalance`, `NivelVinculacion`, `VidaFinanciera` (con percentiles por género desde JSON) |
| 5. EDA | Análisis de churn por geografía, edad, productos, actividad y correlaciones, orientado a preguntas de negocio |
| 6. Preparación para modelado | Split estratificado 70/15/15 (train/val/test), escalado con StandardScaler sin data leakage |
| 7. Modelado | Regresión Logística (interpretable) y Random Forest (mayor desempeño), ambos con `class_weight='balanced'` |
| 8. Métricas y evaluación | AUC-ROC, KS, precision, recall, F1, matriz de confusión, análisis de umbral (0.3 sugerido por costos de negocio) |
| 9. Estabilidad (PSI) | Population Stability Index entre train y test para validar que el modelo no presenta drift |
| 10. Interpretabilidad (SHAP) | TreeExplainer sobre Random Forest: beeswarm plot y ranking de variables más influyentes |
| 11. Recomendaciones de negocio | Segmentos prioritarios, acciones comerciales concretas y estimación de valor económico |
| Anexo | Ejemplo de despliegue como API REST con FastAPI (no ejecutable en Colab) |

## Tecnologías

- **Python 3.12** en Google Colab
- **Librerías:** pandas, numpy, matplotlib, seaborn, scikit-learn, shap
- **Formato de datos:** CSV (dataset), JSON (contexto de negocio)

## Modelos

| Modelo | Tipo | Propósito |
|--------|------|-----------|
| Regresión Logística | Interpretable (baseline) | Coeficientes directos, explicable al negocio |
| Random Forest | Ensamble (200 árboles, max_depth=10) | Mayor capacidad predictiva, captura no-linealidades |

## Métricas clave

- **AUC-ROC** y **KS** para capacidad discriminativa
- **Precision / Recall / F1** con análisis de umbral
- **PSI < 0.10** para estabilidad del modelo
- **Umbral de clasificación: 0.3** — prioriza recall porque perder un cliente ($500) cuesta 6x más que una campaña de retención innecesaria ($80)

## Cómo ejecutar

1. Abrir `curn_model.ipynb` en Google Colab
2. Subir el archivo `Churn_Modelling.csv` a `/content/`
3. Ejecutar todas las celdas en orden secuencial (de arriba hacia abajo)

## Autor

Prueba técnica — Analista Jr. de Ciencia de Datos
