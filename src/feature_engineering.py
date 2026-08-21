"""
feature_engineering.py
Tennis Performance Intelligence

Funciones reutilizables de construcción de features para el modelo
predictivo, extraídas del Notebook 04_machine_learning.ipynb. Incluye una
barrera estructural anti-leakage basada en column_classification.json
(generado en el Notebook 1).

Uso típico:
    from src.feature_engineering import build_pre_match_features, get_feature_columns

    df_feat = build_pre_match_features(df_clean)
    feature_cols = get_feature_columns("data/processed/column_classification.json")
    X = df_feat[feature_cols]
"""

import json
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

# Features candidatas del proyecto (deben estar disponibles ANTES del partido)
CANDIDATE_FEATURE_COLS = [
    "rank_diff", "points_diff", "odds_diff", "has_odds",
    "home_rank", "away_rank",
    "surface", "tour_type_human", "round",
]


def build_pre_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye variables derivadas disponibles antes del partido (sin leakage):

    - rank_diff:   diferencia de ranking oficial (home - away)
    - points_diff: diferencia de puntos de ranking (home - away), nulos -> 0
    - odds_diff:   diferencia de odds de casas de apuestas (home - away)
    - has_odds:    flag binario, 1 si el partido tiene odds registradas

    El flag `has_odds` permite al modelo distinguir "diferencia de odds = 0
    real" de "no había información de odds" tras la imputación con 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio con columnas home_rank, away_rank, home_points,
        away_points, home_odds_match_winner, away_odds_match_winner.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con las 4 columnas derivadas añadidas.
    """
    df = df.copy()

    df["rank_diff"] = df["home_rank"] - df["away_rank"]
    df["points_diff"] = (df["home_points"] - df["away_points"]).fillna(0)

    df["has_odds"] = df["home_odds_match_winner"].notna().astype(int)
    df["odds_diff"] = (
        df["home_odds_match_winner"] - df["away_odds_match_winner"]
    ).fillna(0)

    return df


def load_leakage_rules(json_path: Union[str, Path]) -> dict:
    """
    Carga las reglas de clasificación de columnas generadas en el Notebook 1
    (target, columnas de leakage, features pre-partido sugeridas).

    Parameters
    ----------
    json_path : str o Path
        Ruta a column_classification.json.

    Returns
    -------
    dict
        Diccionario con las claves "target", "leakage_cols", "pre_match_features".
    """
    with open(json_path) as f:
        return json.load(f)


def get_feature_columns(
    json_path: Union[str, Path],
    candidate_cols: List[str] = None,
) -> List[str]:
    """
    Devuelve la lista final de features a usar en el modelo, verificando
    estructuralmente que ninguna columna candidata sea una columna de
    leakage (resultado del partido, no disponible antes de jugarse).

    Esta función es la "barrera anti-leakage" del proyecto: si alguien
    agrega por error una columna prohibida a `candidate_cols`, la función
    lanza un AssertionError en vez de permitir que el error pase silencioso
    al entrenamiento del modelo.

    Parameters
    ----------
    json_path : str o Path
        Ruta a column_classification.json.
    candidate_cols : list de str, opcional
        Lista de columnas candidatas a usar como features. Si no se
        especifica, usa CANDIDATE_FEATURE_COLS (las features definidas en
        el Notebook 4).

    Returns
    -------
    list de str
        Lista de columnas verificadas, seguras de usar como features.

    Raises
    ------
    AssertionError
        Si alguna columna candidata está en la lista de leakage.
    """
    rules = load_leakage_rules(json_path)
    leakage_cols = set(rules["leakage_cols"])

    cols = candidate_cols if candidate_cols is not None else CANDIDATE_FEATURE_COLS

    violations = [c for c in cols if c in leakage_cols]
    assert not violations, (
        f"¡LEAKAGE DETECTADO! Las siguientes columnas son resultado del "
        f"partido y no deben usarse como features: {violations}"
    )

    return cols


def get_home_win_target(df: pd.DataFrame, winner_col: str = "winner_code") -> pd.Series:
    """
    Construye el target binario `home_win` a partir de winner_code.

    winner_code == 1 -> gana home -> home_win = 1
    winner_code == 2 -> gana away -> home_win = 0

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna winner_code.
    winner_col : str
        Nombre de la columna que codifica el ganador (default: "winner_code").

    Returns
    -------
    pd.Series
        Serie binaria (0/1), mismo índice que df.
    """
    return (df[winner_col] == 1).astype(int)
