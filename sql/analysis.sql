-- ============================================================================
-- analysis.sql
-- Tennis Performance Intelligence — Análisis SQL
-- ============================================================================
-- Motor: DuckDB (lee el .parquet directamente, sin necesidad de importar datos
-- a un motor de base de datos tradicional). Sintaxis compatible en su mayoría
-- con PostgreSQL — ver notas de compatibilidad al final del archivo.
--
-- Input: data/processed/matches_clean.parquet
-- Autor: Juan Felipe Lozano
-- ============================================================================


-- ============================================================================
-- SETUP: Vistas base
-- ============================================================================

-- Vista principal: partidos finalizados únicamente (consistente con los
-- notebooks de EDA y estadística, que excluyen partidos no jugados)
CREATE OR REPLACE VIEW matches AS
SELECT *
FROM read_parquet('data/processed/matches_clean.parquet')
WHERE status = 'FINISHED';

-- Vista derivada: reestructura cada partido en dos filas (una por jugador),
-- igual que se hizo en pandas en el Notebook 2. Esta vista funciona como
-- una segunda "tabla" real para poder hacer JOINs genuinos, dado que el
-- dataset fuente es una única tabla plana de partidos.
CREATE OR REPLACE VIEW player_matches AS
SELECT
    unique_match_id,
    home_name   AS player,
    away_name   AS opponent,
    CASE WHEN winner_code = 1 THEN 1 ELSE 0 END AS won,
    home_rank   AS player_rank,
    away_rank   AS opponent_rank,
    surface,
    tour_type_human,
    season_year,
    match_date
FROM matches
UNION ALL
SELECT
    unique_match_id,
    away_name   AS player,
    home_name   AS opponent,
    CASE WHEN winner_code = 2 THEN 1 ELSE 0 END AS won,
    away_rank   AS player_rank,
    home_rank   AS opponent_rank,
    surface,
    tour_type_human,
    season_year,
    match_date
FROM matches;


-- ============================================================================
-- Q1 — ¿Cuántos partidos se jugaron por temporada y circuito?
-- Técnicas: SELECT/WHERE, GROUP BY, ORDER BY
-- ============================================================================
SELECT
    season_year,
    tour_type_human,
    COUNT(*) AS total_partidos
FROM matches
WHERE season_year BETWEEN 2023 AND 2026
GROUP BY season_year, tour_type_human
ORDER BY season_year, tour_type_human;


-- ============================================================================
-- Q2 — ¿Qué jugadores tienen el mejor win rate? (mínimo 30 partidos jugados)
-- Técnicas: GROUP BY, HAVING, ORDER BY
-- ============================================================================
SELECT
    player,
    COUNT(*)                              AS partidos_jugados,
    SUM(won)                              AS partidos_ganados,
    ROUND(100.0 * SUM(won) / COUNT(*), 1) AS win_rate_pct
FROM player_matches
GROUP BY player
HAVING COUNT(*) >= 30
ORDER BY win_rate_pct DESC
LIMIT 15;


-- ============================================================================
-- Q3 — ¿Cómo varía el win rate del favorito (mejor ranking) según superficie?
-- Técnicas: CASE, GROUP BY, ORDER BY
-- ============================================================================
SELECT
    surface,
    COUNT(*) AS partidos_con_ranking,
    ROUND(100.0 * SUM(
        CASE
            WHEN home_rank < away_rank AND winner_code = 1 THEN 1
            WHEN away_rank < home_rank AND winner_code = 2 THEN 1
            ELSE 0
        END
    ) / COUNT(*), 1) AS pct_victorias_favorito
FROM matches
WHERE home_rank IS NOT NULL AND away_rank IS NOT NULL
GROUP BY surface
ORDER BY pct_victorias_favorito DESC;


-- ============================================================================
-- Q4 — Clasificar cada partido según la magnitud de la diferencia de ranking
-- Técnicas: CASE, GROUP BY, ORDER BY
-- ============================================================================
SELECT
    CASE
        WHEN ABS(home_rank - away_rank) <= 10  THEN '01. Muy parejo (0-10)'
        WHEN ABS(home_rank - away_rank) <= 50  THEN '02. Parejo (11-50)'
        WHEN ABS(home_rank - away_rank) <= 100 THEN '03. Moderado (51-100)'
        WHEN ABS(home_rank - away_rank) <= 500 THEN '04. Amplio (101-500)'
        ELSE '05. Muy amplio (500+)'
    END AS categoria_diferencia_ranking,
    COUNT(*) AS total_partidos
FROM matches
WHERE home_rank IS NOT NULL AND away_rank IS NOT NULL
GROUP BY categoria_diferencia_ranking
ORDER BY categoria_diferencia_ranking;


-- ============================================================================
-- Q5 — Head-to-head entre dos jugadores específicos (ejemplo: Sinner vs Zverev)
-- Técnicas: JOIN (self-join sobre matches), CASE, WHERE
-- ============================================================================
SELECT
    m.season_year,
    m.tournament,
    m.round,
    m.surface,
    m.home_name,
    m.away_name,
    CASE
        WHEN m.winner_code = 1 THEN m.home_name
        WHEN m.winner_code = 2 THEN m.away_name
        ELSE 'N/A'
    END AS ganador
FROM matches m
WHERE (m.home_name = 'Sinner J.' AND m.away_name = 'Zverev A.')
   OR (m.home_name = 'Zverev A.' AND m.away_name = 'Sinner J.')
ORDER BY m.match_date;


-- ============================================================================
-- Q6 — Comparar el ranking promedio de un jugador contra sus oponentes
-- (JOIN entre player_matches y una agregación de sí misma)
-- Técnicas: JOIN, subquery, GROUP BY
-- ============================================================================
SELECT
    pm.player,
    ROUND(AVG(pm.player_rank), 1)   AS ranking_promedio_propio,
    ROUND(AVG(pm.opponent_rank), 1) AS ranking_promedio_rivales,
    agg.win_rate_pct
FROM player_matches pm
JOIN (
    SELECT
        player,
        ROUND(100.0 * SUM(won) / COUNT(*), 1) AS win_rate_pct
    FROM player_matches
    GROUP BY player
    HAVING COUNT(*) >= 30
) agg ON pm.player = agg.player
WHERE pm.player_rank IS NOT NULL AND pm.opponent_rank IS NOT NULL
GROUP BY pm.player, agg.win_rate_pct
ORDER BY agg.win_rate_pct DESC
LIMIT 15;


-- ============================================================================
-- Q7 — Jugadores con win rate por ENCIMA del promedio general
-- Técnicas: Subquery (en HAVING)
-- ============================================================================
SELECT
    player,
    COUNT(*)                              AS partidos_jugados,
    ROUND(100.0 * SUM(won) / COUNT(*), 1) AS win_rate_pct
FROM player_matches
GROUP BY player
HAVING COUNT(*) >= 30
   AND (100.0 * SUM(won) / COUNT(*)) > (
        SELECT 100.0 * SUM(won) / COUNT(*) FROM player_matches
   )
ORDER BY win_rate_pct DESC;


-- ============================================================================
-- Q8 — "Upsets": partidos donde ganó el jugador con PEOR ranking (sorpresas)
-- Técnicas: Subquery (correlacionada), WHERE, CASE
-- ============================================================================
SELECT
    m.season_year,
    m.tournament,
    m.surface,
    m.home_name, m.home_rank,
    m.away_name, m.away_rank,
    CASE WHEN m.winner_code = 1 THEN m.home_name ELSE m.away_name END AS ganador,
    ABS(m.home_rank - m.away_rank) AS diferencia_ranking
FROM matches m
WHERE m.home_rank IS NOT NULL AND m.away_rank IS NOT NULL
  AND (
        (m.winner_code = 1 AND m.home_rank > m.away_rank)  -- ganó home siendo peor rankeado
     OR (m.winner_code = 2 AND m.away_rank > m.home_rank)  -- ganó away siendo peor rankeado
      )
  AND ABS(m.home_rank - m.away_rank) > (
        SELECT AVG(ABS(home_rank - away_rank))
        FROM matches
        WHERE home_rank IS NOT NULL AND away_rank IS NOT NULL
      )
ORDER BY diferencia_ranking DESC
LIMIT 20;


-- ============================================================================
-- Q9 — Win rate del favorito por bucket de diferencia de ranking (CTE)
-- Técnicas: CTE (WITH), CASE, GROUP BY
-- ============================================================================
WITH partidos_con_bucket AS (
    SELECT
        CASE
            WHEN ABS(home_rank - away_rank) <= 10  THEN '01. 0-10'
            WHEN ABS(home_rank - away_rank) <= 50  THEN '02. 11-50'
            WHEN ABS(home_rank - away_rank) <= 100 THEN '03. 51-100'
            WHEN ABS(home_rank - away_rank) <= 500 THEN '04. 101-500'
            ELSE '05. 500+'
        END AS bucket,
        CASE
            WHEN home_rank < away_rank AND winner_code = 1 THEN 1
            WHEN away_rank < home_rank AND winner_code = 2 THEN 1
            ELSE 0
        END AS gano_favorito
    FROM matches
    WHERE home_rank IS NOT NULL AND away_rank IS NOT NULL
)
SELECT
    bucket,
    COUNT(*)                                  AS total_partidos,
    ROUND(100.0 * SUM(gano_favorito) / COUNT(*), 1) AS pct_victorias_favorito
FROM partidos_con_bucket
GROUP BY bucket
ORDER BY bucket;


-- ============================================================================
-- Q10 — Ranking de jugadores por número de victorias dentro de cada temporada
-- Técnicas: Window function RANK()
-- ============================================================================
SELECT *
FROM (
    SELECT
        season_year,
        player,
        SUM(won) AS victorias,
        RANK() OVER (PARTITION BY season_year ORDER BY SUM(won) DESC) AS ranking_victorias
    FROM player_matches
    GROUP BY season_year, player
) ranked
WHERE ranking_victorias <= 5
ORDER BY season_year, ranking_victorias;


-- ============================================================================
-- Q11 — Partido más reciente jugado por cada jugador
-- Técnicas: Window function ROW_NUMBER()
-- ============================================================================
SELECT player, opponent, surface, match_date, won
FROM (
    SELECT
        player, opponent, surface, match_date, won,
        ROW_NUMBER() OVER (PARTITION BY player ORDER BY match_date DESC) AS rn
    FROM player_matches
) recientes
WHERE rn = 1
ORDER BY match_date DESC
LIMIT 20;


-- ============================================================================
-- Q12 — Evolución del ranking de un jugador entre partidos consecutivos
-- Técnicas: Window functions LAG/LEAD
-- ============================================================================
SELECT
    player,
    match_date,
    player_rank,
    LAG(player_rank)  OVER (PARTITION BY player ORDER BY match_date) AS ranking_partido_anterior,
    LEAD(player_rank) OVER (PARTITION BY player ORDER BY match_date) AS ranking_partido_siguiente,
    player_rank - LAG(player_rank) OVER (PARTITION BY player ORDER BY match_date) AS variacion_ranking
FROM player_matches
WHERE player = 'Sinner J.'
  AND player_rank IS NOT NULL
ORDER BY match_date;


-- ============================================================================
-- NOTAS DE COMPATIBILIDAD
-- ============================================================================
-- - read_parquet() es específico de DuckDB. En PostgreSQL, reemplazar la
--   vista `matches` por una tabla importada previamente vía COPY o un ETL
--   externo (ej. pandas.to_sql).
-- - RANK(), ROW_NUMBER(), LAG(), LEAD() son estándar ANSI SQL y funcionan
--   igual en DuckDB, PostgreSQL, SQL Server y SQLite (3.25+).
-- - UNION ALL, CTEs (WITH) y subqueries correlacionadas son estándar y
--   portables sin cambios.
-- ============================================================================
