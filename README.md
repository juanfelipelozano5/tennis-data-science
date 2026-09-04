# Tennis Performance Intelligence

Análisis y predicción de resultados de tenis profesional (ATP/WTA, 2023–2026)
mediante un pipeline completo de Data Science: limpieza y auditoría de datos,
análisis exploratorio, inferencia estadística, machine learning y un dashboard
interactivo en HTML.

**Prueba técnica — Data Analyst / Junior Data Scientist**
**Autor:** Juan Felipe Lozano

---

## Tabla de contenido

1. [Problema](#1-problema)
2. [Dataset](#2-dataset)
3. [Metodología](#3-metodología)
4. [Data Cleaning](#4-data-cleaning)
5. [EDA](#5-eda-análisis-exploratorio)
6. [Estadística](#6-estadística)
7. [Machine Learning](#7-machine-learning)
8. [Resultados](#8-resultados)
9. [Insights](#9-insights-principales)
10. [Limitaciones](#10-limitaciones)
11. [Tecnologías](#11-tecnologías)
12. [Reproducción del proyecto](#12-reproducción-del-proyecto)

---

## 1. Problema

**Tennis Performance Intelligence** necesita entender qué factores están
asociados al resultado de un partido de tenis profesional y evaluar si es
posible predecir al ganador **antes de que el partido se juegue**, usando
únicamente información disponible en ese momento (sin variables resultado del
partido).

**Pregunta principal:**
> ¿Qué factores explican mejor la probabilidad de ganar un partido de tenis
> y qué tan bien podemos predecir su resultado antes de que se juegue?

Se investigan además diferencias por circuito (ATP/WTA), superficie, jugador,
torneo y temporada.

---

## 2. Dataset

- **Fuente:** 8 archivos CSV con partidos ATP y WTA por temporada (2023–2026).
- **Volumen:** 74,703 partidos totales; 74,578 finalizados (`status = FINISHED`)
  tras excluir 125 partidos programados sin jugar (0.2%).
- **Columnas:** 49 columnas originales por partido (identificadores,
  torneo, ranking, sets, estadísticas de servicio/resto, odds de casas de
  apuestas), más columnas derivadas (`unique_match_id`, `match_date`,
  `source_file`).
- **Estructura:** una fila por partido, con estadísticas de ambos jugadores
  bajo prefijos `home_`/`away_` — nomenclatura heredada de la fuente, **sin**
  significado de local/visitante (no aplica en tenis individual).

El **data dictionary** completo, generado automáticamente a partir del
dataset limpio, está en
[`data/processed/data_dictionary.csv`](data/processed/data_dictionary.csv).

### Hallazgo de auditoría: `match_id` no es una llave única

Se detectaron 11 casos de `match_id` repetido correspondientes a partidos
completamente distintos (diferente tour, temporada y jugadores) —
colisión de IDs entre los archivos fuente de ATP y WTA. Se corrigió creando
`unique_match_id` (`match_id + season_year + tour_type`) como llave primaria
real del proyecto, usada consistentemente en notebooks y SQL.

---

## 3. Metodología

Pipeline estructurado en 4 notebooks independientes (cada uno reproducible
desde `data/processed/matches_clean.parquet`, sin depender de estado en
memoria de otros notebooks), más un análisis SQL y un dashboard interactivo
en HTML/JavaScript (Chart.js):

```
01_data_loading_cleaning.ipynb  →  02_eda.ipynb  →  03_statistics_hypothesis.ipynb  →  04_machine_learning.ipynb
        ↓                                                                                        ↑
   data/processed/matches_clean.parquet ───────────────────────────────────────────────────────┘
        ↓
   sql/analysis.sql (DuckDB, lee el .parquet directamente)
```

Decisión de diseño clave: la función `clean_matches()` (en
[`src/data_loading.py`](src/data_loading.py)) centraliza toda la lógica de
limpieza, de forma que el proceso es reproducible y testeable fuera de un
notebook.

---

## 4. Data Cleaning

Auditoría completa en [`notebooks/01_data_loading_cleaning.ipynb`](notebooks/01_data_loading_cleaning.ipynb):

- **Tipos de datos:** verificados y forzados a numérico en columnas de
  estadísticas (`*_perc`, `*_points`, `*_faults`).
- **Nulos:**
  - Nulidad **estructural** (no requiere imputación): `set_3/4/5_score`
    (65%–99% nulos) — partidos decididos en menos sets del máximo posible.
  - Nulidad por **cobertura de la fuente**: `home_rank`/`away_rank`/`*_points`
    (~44% nulos, jugadores sin ranking oficial); `*_odds_*` (3.5%–5% nulos,
    partidos sin casa de apuestas registrada).
- **Duplicados:** 0 duplicados exactos. 11 casos de `match_id` repetido
  (ver sección 2) resueltos con `unique_match_id`.
- **Categorías:** normalización de texto (`surface`, `round`,
  `tour_type_human`). Se reclasificó la categoría `"No Surface"` (894
  partidos, 1.2%) como `"Team Event (Mixed Surface)"` — corresponde en su
  totalidad a partidos de Davis Cup (ATP) y Billie Jean King Cup (WTA), donde
  la superficie varía por eliminatoria según el país anfitrión.
- **Outliers:** revisados visualmente (boxplots) en variables de servicio;
  sin evidencia de outliers no plausibles (todos los valores caen en rangos
  posibles para un partido de tenis).
- **Calidad ATP vs. WTA:** patrones de missingness comparados por circuito;
  documentado en el notebook.

---

## 5. EDA (Análisis Exploratorio)

Desarrollado en [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb).

### Insights descriptivos

1. ATP concentra mayor volumen de partidos que WTA (74,578 partidos
   finalizados analizados).
2. La distribución de superficie varía por circuito: los tours principales
   (ATP Tour, WTA Tour) están dominados por hard court (52.2% y 58.3%
   respectivamente), mientras los circuitos Challenger se inclinan más
   hacia clay — especialmente WTA Chall (62.0% clay vs. 32.6% hard).
3. El favorito (mejor ranking) gana el **62.6%** de los partidos analizados
   (n=40,760 con ranking disponible en ambos jugadores).
4. Los ganadores promedian **66.05%** de puntos de servicio ganados vs.
   **53.95%** de los perdedores, y 4.54 aces vs. 3.30.
5. Entre jugadores con ≥30 partidos, Sinner J. (87.8%) y Swiatek I. (81.7%)
   lideran el win rate del período analizado.

### Insights analíticos

1. La probabilidad de victoria del favorito aumenta **monotónicamente** con
   la diferencia de ranking: de 53.6% (diferencia 0–10 puestos) a 77.0%
   (diferencia 500+ puestos).
2. El dominio en estadísticas de servicio (+12.1 pp en % de puntos ganados)
   caracteriza consistentemente a los partidos ganados.
3. Correlación fuerte (**r = -0.78**) entre servicio y resto del mismo
   jugador — la más alta de la matriz de correlación; sin multicolinealidad
   severa entre variables candidatas a features pre-partido.

---

## 6. Estadística

Desarrollado en [`notebooks/03_statistics_hypothesis.ipynb`](notebooks/03_statistics_hypothesis.ipynb).

### Hipótesis 1 — ¿Gana el favorito más que por azar?

| | |
|---|---|
| H0 | La probabilidad de que gane el favorito es igual a 0.5 |
| H1 | La probabilidad de que gane el favorito es distinta de 0.5 |
| Prueba | Test de proporción de una muestra (z-test) |
| α | 0.05 |
| Estadístico | z = 52.46 |
| p-value | ≈ 0 |
| **Conclusión** | **Se rechaza H0.** El favorito gana significativamente más que el 50% esperado por azar (observado: 62.6%, IC 95% bootstrap: [62.14%, 63.07%]). |

### Hipótesis 2 — ¿Difiere el servicio entre ganadores y perdedores?

| | |
|---|---|
| H0 | No hay diferencia en % de servicio ganado entre ganadores y perdedores |
| H1 | Sí hay diferencia |
| Prueba | t-test de muestras pareadas |
| α | 0.05 |
| Estadístico | t = 329.31 |
| p-value | ≈ 0 |
| **Conclusión** | **Se rechaza H0.** Diferencia media de 12.11 puntos porcentuales (66.05% vs. 53.95%). |

### Correlación vs. causalidad

Se discute explícitamente que las asociaciones encontradas (ranking,
servicio) son observacionales y estadísticamente robustas, pero no permiten
afirmar causalidad aislada sin un diseño experimental — el ranking es en sí
mismo una medida acumulada de rendimiento pasado, y el dominio del servicio
podría reflejar un tercer factor no observado (nivel general del jugador ese
día).

---

## 7. Machine Learning

Desarrollado en [`notebooks/04_machine_learning.ipynb`](notebooks/04_machine_learning.ipynb).

### Prevención de data leakage

Las features se seleccionan mediante una barrera estructural
(`get_feature_columns()` en [`src/feature_engineering.py`](src/feature_engineering.py))
que verifica con un `assert` que ninguna columna candidata esté en la lista
de columnas de leakage documentada en `data/processed/column_classification.json`
(generada en el Notebook 1). Esto convierte la prevención de leakage en una
regla de código verificable, no solo en una intención documentada.

**Features utilizadas (pre-partido):** `rank_diff`, `points_diff`,
`odds_diff`, `has_odds`, `home_rank`, `away_rank`, `surface`,
`tour_type_human`, `round`.

**Excluidas por leakage:** todas las estadísticas resultado del partido —
sets, aces, dobles faltas, % de servicio/resto/break points (24 columnas).

### Modelos comparados

| Modelo | Accuracy (test) | ROC-AUC (test) |
|---|---|---|
| Baseline (clase mayoritaria) | 51.2% | 0.50 |
| **Logistic Regression** | **67.4%** | **0.7374** |
| Decision Tree | 67.1% | 0.7356 |
| Random Forest (base) | 67.0% | 0.7311 |
| Random Forest (tuned, Grid Search) | 66.9% | 0.7334 |

### Selección del modelo final

Se optimizó Random Forest mediante Grid Search (`n_estimators`, `max_depth`,
`min_samples_leaf`, scoring por ROC-AUC), pero **Logistic Regression sin
ajustar terminó superando al Random Forest optimizado** en el test set. En
cross-validation los tres modelos base ya quedaban muy cerca entre sí
(diferencias dentro del margen de ruido, ±0.003), lo que sugiere que la
relación entre las features y el resultado es mayormente lineal — un
modelo simple captura casi toda la señal disponible.

**Se selecciona Logistic Regression como modelo final**: iguala o supera a
Random Forest, con menos hiperparámetros, mayor velocidad de entrenamiento y
coeficientes directamente interpretables. Random Forest se conserva como
modelo de referencia para el análisis de feature importance.

### Feature importance

La variable más influyente con amplio margen es `odds_diff` (49% de
importancia en Random Forest) — el modelo aprende en gran medida a confiar
en las odds de casas de apuestas, que ya incorporan información experta y
de mercado. Le siguen `points_diff` (17%), `rank_diff` (14%), y
`home_rank`/`away_rank` (~7% cada uno).

### Overfitting y generalización

Gap train-test de 0.0158 (accuracy) — pequeño, sin señales de overfitting
severo.

### ¿El modelo es fiable para uso real?

**Parcialmente.** Es fiable como **herramienta de apoyo analítico** —
supera claramente al baseline y al azar, y generaliza razonablemente bien —
pero con ROC-AUC≈0.74 el modelo acierta el ganador en aproximadamente 2 de
cada 3 partidos, un margen de error considerable para cualquier decisión de
alto riesgo. No debería usarse como sistema de decisión autónomo,
especialmente porque su feature más influyente (`odds_diff`) ya incorpora
información de mercado — el modelo sistematiza esa información más que
aportar una ventaja informativa nueva sobre ella.

---

## 8. Resultados

- **74,578 partidos** auditados y limpiados; **40,760 partidos** utilizados
  en el modelo predictivo (con ranking disponible en ambos jugadores).
- **Dos hipótesis validadas** con significancia estadística robusta
  (p ≈ 0 en ambas).
- **Modelo final: Logistic Regression**, ROC-AUC = 0.7374, superando al
  baseline en +16.2 puntos porcentuales de accuracy.
- **12 preguntas de negocio** resueltas en SQL ([`sql/analysis.sql`](sql/analysis.sql)),
  cubriendo victorias, win rate, ranking, superficies, circuitos, evolución
  temporal, head-to-head y detección de "upsets".

---

## 9. Insights principales

1. El ranking oficial es un predictor real y significativo, con relación
   monotónica clara respecto a la probabilidad de victoria.
2. El dominio del servicio caracteriza fuertemente las victorias, pero es
   resultado del partido (no utilizable como feature predictiva pre-partido).
3. Las odds de mercado (`odds_diff`) son, por sí solas, la señal más potente
   disponible antes del partido — más que el ranking oficial.
4. La categoría "Team Event" (Davis Cup / BJK Cup) es un hallazgo de
   auditoría de datos genuino, no un error — documentado y tratado
   explícitamente en el pipeline.
5. Un modelo lineal simple (Logistic Regression) iguala a modelos más
   complejos — la señal disponible en los datos actuales no tiene
   suficiente no-linealidad para justificar mayor complejidad.

---

## 10. Limitaciones

- **Cobertura parcial del modelo:** solo predice partidos con ranking
  disponible en ambos jugadores (~55% del total de partidos finalizados) —
  excluye estructuralmente qualifiers y wildcards sin ranking oficial.
- **Sin historial reciente ni head-to-head como features:** no se
  incorporó racha de victorias reciente ni historial de enfrentamientos
  directos al modelo predictivo — extensión natural para trabajo futuro.
- **Datos de odds incompletos:** ~3.5–5% de partidos sin odds registradas,
  tratado con el flag `has_odds`, pero introduce cierto ruido.
- **Naturaleza del deporte:** el tenis tiene componentes de variabilidad
  día a día (forma física, lesiones, condiciones) que ninguna feature
  pre-partido puede capturar completamente — el techo de ROC-AUC≈0.74 con
  las features actuales es consistente con esta limitación inherente.
- **Correlación, no causalidad:** las asociaciones estadísticas encontradas
  (ranking, servicio) no implican relaciones causales aisladas.

---

## 11. Tecnologías

| Categoría | Herramientas |
|---|---|
| Lenguaje | Python 3.10+ |
| Manipulación de datos | pandas, numpy |
| Visualización | matplotlib, seaborn |
| Estadística | scipy, statsmodels |
| Machine Learning | scikit-learn |
| SQL | DuckDB (lectura directa de Parquet) |
| Control de versiones | Git, GitHub |
| Dashboard | HTML, CSS, JavaScript, Chart.js |
| Entorno de desarrollo | Google Colab |

---

## 12. Reproducción del proyecto

### Estructura del repositorio

```
tennis-data-science/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # 8 CSV originales (ATP/WTA, 2023-2026)
│   └── processed/              # matches_clean.parquet, data_dictionary.csv,
│                                # column_classification.json
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_statistics_hypothesis.ipynb
│   └── 04_machine_learning.ipynb
├── sql/
│   └── analysis.sql            # 12 preguntas de negocio (DuckDB)
├── src/
│   ├── data_loading.py         # carga y limpieza reutilizable
│   ├── feature_engineering.py  # features + barrera anti-leakage
│   └── player_utils.py         # análisis a nivel de jugador
├── dashboard/
│   ├── index.html               # dashboard interactivo (abrir en el navegador)
│   └── data/                    # CSV agregados usados como insumo del dashboard
└── images/                     # gráficos clave exportados de los notebooks
```

### Pasos para reproducir

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/juanfelipelozano5/tennis-data-science.git
   cd tennis-data-science
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el pipeline de limpieza** (genera `data/processed/`):
   ```bash
   python src/data_loading.py
   ```
   O, alternativamente, correr `notebooks/01_data_loading_cleaning.ipynb`
   de principio a fin.

4. **Ejecutar los notebooks en orden** (cada uno lee desde
   `data/processed/matches_clean.parquet`, ninguno depende de estado en
   memoria de otro):
   ```
   notebooks/02_eda.ipynb
   notebooks/03_statistics_hypothesis.ipynb
   notebooks/04_machine_learning.ipynb
   ```

5. **Ejecutar el análisis SQL:**
   ```python
   import duckdb
   con = duckdb.connect()
   con.execute(open("sql/analysis.sql").read())
   ```

6. **Abrir el dashboard:** `dashboard/index.html` directamente en cualquier
   navegador (no requiere servidor ni instalación — usa Chart.js vía CDN).

### Notas de reproducibilidad

- Todos los notebooks fueron verificados con "Restart & Run All" para
  garantizar que corren de principio a fin sin dependencias de estado
  manual previo.
- La semilla aleatoria (`RANDOM_STATE = 42`) se mantiene fija en todo el
  pipeline de Machine Learning para reproducibilidad exacta de resultados.
