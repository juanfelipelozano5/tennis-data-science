"""
data_loading.py
Tennis Performance Intelligence

Funciones reutilizables de carga y limpieza de datos, extraídas del
Notebook 01_data_loading_cleaning.ipynb. Permiten reproducir el proceso de
consolidación y limpieza fuera del entorno de notebook (scripts, pipelines,
tests automatizados).

Uso típico:
    from src.data_loading import load_raw_matches, clean_matches

    df_raw = load_raw_matches("data/raw")
    df_clean = clean_matches(df_raw)
    df_clean.to_parquet("data/processed/matches_clean.parquet", index=False)
"""

from pathlib import Path
from typing import Union

import pandas as pd

# Archivos fuente esperados en data/raw/
RAW_FILES = [
    "2023-atp-season.csv",
    "2023-wta-season.csv",
    "2024-atp-season.csv",
    "2024-wta-season.csv",
    "2025-atp-season.csv",
    "2025-wta-season.csv",
    "2026-atp-season.csv",
    "2026-wta-season.csv",
]


def load_raw_matches(data_dir: Union[str, Path] = "data/raw") -> pd.DataFrame:
    """
    Carga y consolida los 8 archivos CSV de temporadas ATP/WTA (2023-2026).

    Añade la columna `source_file` para trazabilidad de cada fila hacia su
    archivo de origen — útil para auditorías de calidad por temporada/circuito.

    Parameters
    ----------
    data_dir : str o Path
        Carpeta donde están los 8 archivos CSV originales.

    Returns
    -------
    pd.DataFrame
        DataFrame consolidado con todos los partidos y la columna source_file.
    """
    data_dir = Path(data_dir)
    dfs = []
    for filename in RAW_FILES:
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"No se encontró {filepath}. Verifica que los 8 CSV estén en {data_dir}."
            )
        tmp = pd.read_csv(filepath)
        tmp["source_file"] = filename
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)
    return df


def add_unique_match_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una llave primaria única y confiable para cada partido.

    `match_id` por sí solo NO es único globalmente: la fuente de datos
    reutiliza el mismo espacio de IDs entre ATP/WTA y entre temporadas
    distintas (hallazgo documentado en el Notebook 1, auditoría de
    duplicados). Se combina con `season_year` y `tour_type` para garantizar
    unicidad real.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas match_id, season_year, tour_type.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con la columna `unique_match_id` añadida.
    """
    df = df.copy()
    df["unique_match_id"] = (
        df["match_id"].astype(str)
        + "_"
        + df["season_year"].astype(str)
        + "_"
        + df["tour_type"].astype(str)
    )
    return df


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline de limpieza estándar del proyecto:

    1. Normaliza texto categórico (surface, round, tour_type_human).
    2. Reclasifica la categoría "No Surface" como "Team Event (Mixed Surface)"
       — corresponde a partidos de Davis Cup / Billie Jean King Cup, donde
       la superficie varía por eliminatoria según el país anfitrión y no se
       registra de forma desagregada en la fuente (~1.2% del dataset).
    3. Elimina filas sin match_id válido.
    4. Elimina duplicados exactos.
    5. Fuerza tipos numéricos en columnas de estadísticas de partido.
    6. Añade `unique_match_id` (ver add_unique_match_id) y `match_date`
       (conversión de date_timestamp unix a datetime).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame crudo consolidado (output de load_raw_matches).

    Returns
    -------
    pd.DataFrame
        DataFrame limpio, listo para exportar a data/processed/.
    """
    df_clean = df.copy()

    # 1. Normalizar texto categórico
    for col in ["surface", "round", "tour_type_human"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()

    # 2. Reclasificar "No Surface" -> "Team Event (Mixed Surface)"
    if "surface" in df_clean.columns:
        df_clean.loc[
            df_clean["surface"] == "No Surface", "surface"
        ] = "Team Event (Mixed Surface)"

    # 3. Eliminar filas sin match_id válido
    df_clean = df_clean.dropna(subset=["match_id"])

    # 4. Eliminar duplicados exactos
    df_clean = df_clean.drop_duplicates()

    # 5. Forzar tipos numéricos en columnas de estadísticas
    stat_cols = [
        c for c in df_clean.columns
        if "perc" in c or "points" in c or "faults" in c
    ]
    for col in stat_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # 6. Llave única y fecha legible
    df_clean = add_unique_match_id(df_clean)
    if "date_timestamp" in df_clean.columns:
        df_clean["match_date"] = pd.to_datetime(
            df_clean["date_timestamp"], unit="s", errors="coerce"
        )

    return df_clean


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un data dictionary automático a partir de un DataFrame limpio.

    Incluye tipo de dato, cantidad y porcentaje de nulos, y cardinalidad
    (n_unique) por columna — insumo directo para el README del proyecto.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio (típicamente el output de clean_matches).

    Returns
    -------
    pd.DataFrame
        Data dictionary con una fila por columna.
    """
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "n_nulls": df.isnull().sum().values,
        "pct_nulls": (df.isnull().mean() * 100).round(2).values,
        "n_unique": df.nunique().values,
    })


if __name__ == "__main__":
    # Ejemplo de uso end-to-end, ejecutable como script desde la raíz del repo:
    #   python src/data_loading.py
    raw = load_raw_matches("data/raw")
    clean = clean_matches(raw)
    print(f"Partidos cargados: {len(raw):,}")
    print(f"Partidos tras limpieza: {len(clean):,}")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    clean.to_parquet("data/processed/matches_clean.parquet", index=False)
    build_data_dictionary(clean).to_csv(
        "data/processed/data_dictionary.csv", index=False
    )
    print("Archivos guardados en data/processed/")
