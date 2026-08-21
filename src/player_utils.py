"""
player_utils.py
Tennis Performance Intelligence

Funciones reutilizables para análisis a nivel de jugador, extraídas del
Notebook 02_eda.ipynb. El dataset fuente tiene una fila por PARTIDO
(columnas home/away); estas funciones lo reestructuran a una fila por
JUGADOR-PARTIDO, necesaria para calcular win rate, historial y rendimiento
individual.

Nota de dominio: "home"/"away" en este dataset no equivale a local/visitante
(no aplica en tenis) — es solo el orden en que la fuente listó a los dos
jugadores del partido.

Uso típico:
    from src.player_utils import to_player_long_format, get_player_win_rates

    player_matches = to_player_long_format(df_clean)
    win_rates = get_player_win_rates(player_matches, min_matches=30)
"""

from typing import List, Optional

import pandas as pd


def to_player_long_format(
    df: pd.DataFrame,
    extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Reestructura el dataset de partidos (formato ancho: home/away) a formato
    largo (una fila por jugador-partido).

    Cada partido aporta dos filas de salida: una para el jugador "home" y
    otra para el jugador "away", cada una con su propio resultado (won).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de partidos con columnas home_name, away_name, winner_code,
        unique_match_id, y opcionalmente surface, season_year.
    extra_cols : list de str, opcional
        Columnas adicionales a preservar en el resultado (deben existir en df
        y no depender de home/away, ej. surface, season_year, match_date).

    Returns
    -------
    pd.DataFrame
        DataFrame largo con columnas: unique_match_id, player, won,
        + extra_cols.
    """
    base_cols = ["unique_match_id", "winner_code"]
    extra_cols = extra_cols or []

    home = df[base_cols + ["home_name"] + extra_cols].copy()
    home["player"] = home["home_name"]
    home["won"] = home["winner_code"] == 1

    away = df[base_cols + ["away_name"] + extra_cols].copy()
    away["player"] = away["away_name"]
    away["won"] = away["winner_code"] == 2

    out_cols = ["unique_match_id", "player", "won"] + extra_cols

    player_matches = pd.concat(
        [home[out_cols], away[out_cols]], ignore_index=True
    )
    return player_matches


def get_player_win_rates(
    player_matches: pd.DataFrame,
    min_matches: int = 30,
) -> pd.DataFrame:
    """
    Calcula partidos jugados, ganados y win rate por jugador.

    Aplica un filtro de volumen mínimo de partidos para evitar conclusiones
    estadísticamente ruidosas (ej. un jugador con 2 partidos y 2 victorias
    no debería aparecer con 100% de win rate al tope de un ranking).

    Parameters
    ----------
    player_matches : pd.DataFrame
        Output de to_player_long_format (columnas: player, won).
    min_matches : int
        Número mínimo de partidos jugados para incluir al jugador
        (default: 30, mismo umbral usado en el Notebook 2).

    Returns
    -------
    pd.DataFrame
        DataFrame indexado por player, con columnas matches_played,
        matches_won, win_rate_pct — ordenado de mayor a menor win rate.
    """
    summary = player_matches.groupby("player").agg(
        matches_played=("won", "count"),
        matches_won=("won", "sum"),
    )
    summary["win_rate_pct"] = (
        summary["matches_won"] / summary["matches_played"] * 100
    ).round(1)

    summary = summary[summary["matches_played"] >= min_matches]
    return summary.sort_values("win_rate_pct", ascending=False)


def get_favorite_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula, para cada partido con ranking disponible en ambos jugadores,
    si ganó el jugador favorito (mejor ranking = número más bajo).

    Añade además `rank_diff_bucket`, una categorización de la magnitud de
    la diferencia de ranking — usada en el Notebook 2 y 3 para mostrar que
    la probabilidad de victoria del favorito aumenta con la diferencia.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de partidos con home_rank, away_rank, winner_code.

    Returns
    -------
    pd.DataFrame
        Copia filtrada (solo partidos con ranking en ambos jugadores) con
        las columnas añadidas: favorite_won, rank_diff, rank_diff_bucket.
    """
    df_rank = df.dropna(subset=["home_rank", "away_rank", "winner_code"]).copy()

    df_rank["favorite_won"] = (
        (df_rank["home_rank"] < df_rank["away_rank"]) & (df_rank["winner_code"] == 1)
    ) | (
        (df_rank["away_rank"] < df_rank["home_rank"]) & (df_rank["winner_code"] == 2)
    )

    df_rank["rank_diff"] = (df_rank["home_rank"] - df_rank["away_rank"]).abs()
    df_rank["rank_diff_bucket"] = pd.cut(
        df_rank["rank_diff"],
        bins=[0, 10, 50, 100, 500, float("inf")],
        labels=["0-10", "11-50", "51-100", "101-500", "500+"],
    )

    return df_rank
