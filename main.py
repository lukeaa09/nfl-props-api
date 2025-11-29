from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl
import nflreadpy as nfl

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# -------------------------------------------------------
# Basic config
# -------------------------------------------------------

SEASON = 2025

app = FastAPI(
    title="NFL Props API",
    description=(
        "Simple API that exposes nflverse weekly player stats and "
        "basic matchup-aware projections for educational/analytics use."
    ),
    version="3.0.0",
)

# Allow the Loveable front-end (and others) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
# Global error handler (so the frontend sees JSON, not HTML)
# -------------------------------------------------------


@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": repr(exc),
        },
    )


# -------------------------------------------------------
# Utility: clean NaNs for JSON
# -------------------------------------------------------


def clean_nans(value: Any) -> Any:
    """
    Recursively replace NaN/NaT with None so JSON encoding doesn't break.
    """
    if isinstance(value, float):
        # NaN check: NaN != NaN
        if value != value:
            return None
        return value

    if isinstance(value, dict):
        return {k: clean_nans(v) for k, v in value.items()}

    if isinstance(value, list):
        return [clean_nans(v) for v in value]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


# -------------------------------------------------------
# Load weekly stats & schedule from nflverse
# -------------------------------------------------------


def load_weekly_stats() -> pd.DataFrame:
    """
    Load nflverse weekly player stats for the given season.

    No caching: this always pulls the latest data, so new weeks
    show up automatically when nflverse updates.
    """
    pl_df = nfl.load_player_stats(
        seasons=[SEASON],
        summary_level="week",
    )

    if isinstance(pl_df, pl.DataFrame):
        df = pl_df.to_pandas()
    elif isinstance(pl_df, pd.DataFrame):
        df = pl_df
    else:
        df = pd.DataFrame(pl_df)

    return df


def load_schedule() -> Optional[pd.DataFrame]:
    """
    Load NFL schedule for this season.

    Used to figure out:
      - who each team plays in a given week
      - whether they're home or away
      - whether it's roughly a night/prime-time game
    """
    try:
        pl_df = nfl.load_schedules(seasons=[SEASON])
    except Exception:
        return None

    if isinstance(pl_df, pl.DataFrame):
        df = pl_df.to_pandas()
    elif isinstance(pl_df, pd.DataFrame):
        df = pl_df
    else:
        df = pd.DataFrame(pl_df)

    return df


# -------------------------------------------------------
# Root / health / debug
# -------------------------------------------------------


@app.get("/")
def root():
    """
    Simple root info. Mainly for sanity checks in the browser.
    """
    try:
        df = load_weekly_stats()
        max_week = int(df["week"].max()) if "week" in df.columns else None
    except Exception:
        max_week = None

    return {
        "ok": True,
        "message": "NFL Props API",
        "season": SEASON,
        "current_data_week": max_week,
    }


@app.get("/health")
def health():
    """
    Quick health check used by you and by the frontend.
    """
    try:
        df = load_weekly_stats()
        return {
            "ok": True,
            "season": SEASON,
            "rows": int(len(df)),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/debug_columns")
def debug_columns():
    """
    Debug endpoint: shows columns + a few sample rows.
    Helpful if something breaks.
    """
    try:
        df = load_weekly_stats()
        sample = df.head(3).to_dict(orient="records")
        return {
            "ok": True,
            "columns": list(df.columns),
            "sample": clean_nans(sample),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -------------------------------------------------------
# /players: list/search
# -------------------------------------------------------


@app.get("/players")
def list_players(q: Optional[str] = None):
    """
    List players (id, name, team, position) for offensive positions.
    Optional q: case-insensitive substring match on name.
    """
    df = load_weekly_stats()

    cols = set(df.columns)
    required = {"player_id", "player_display_name", "team", "position"}
    missing = sorted(required - cols)
    if missing:
        return {
            "ok": False,
            "error": "Missing required columns in weekly stats",
            "missing": missing,
            "columns": sorted(cols),
        }

    # Filter to offensive positions
    df_use = df[df["position"].isin(["QB", "RB", "WR", "TE"])]

    base = (
        df_use[["player_id", "player_display_name", "team", "position"]]
        .drop_duplicates()
        .sort_values("player_display_name")
    )

    if q:
        q_lower = q.lower()
        mask = base["player_display_name"].str.lower().str.contains(
            q_lower, na=False
        )
        base = base[mask]

    players = [
        {
            "id": row.player_id,
            "name": row.player_display_name,
            "team": row.team,
            "position": row.position,
        }
        for row in base.itertuples()
    ]

    return {
        "ok": True,
        "count": len(players),
        "players": players,
    }


# -------------------------------------------------------
# /player/{id}: weekly detail
# -------------------------------------------------------


@app.get("/player/{player_id}")
def player_detail(player_id: str, week: Optional[int] = None):
    """
    Return weekly log + overview for a single player.

    Response:
      - player: identity info
      - weeksWithData: which weeks they have stats
      - selectedWeek: which week is currently selected
      - overview: stats row for that week (or latest if week=None)
      - weeklyLog: all weekly rows for that player
    """
    df = load_weekly_stats()

    if "player_id" not in df.columns:
        return {
            "ok": False,
            "error": "No 'player_id' column in data",
            "columns": list(df.columns),
        }

    pdf = df[df["player_id"] == player_id].copy()
    if pdf.empty:
        return {
            "ok": False,
            "error": "Player not found",
        }

    # Weeks with data
    if "week" in pdf.columns:
        weeks_with_data = sorted(int(w) for w in pdf["week"].unique())
    else:
        weeks_with_data = []

    # Decide selected week
    if weeks_with_data:
        if week is None:
            selected_week = max(weeks_with_data)
        else:
            selected_week = int(week)
    else:
        selected_week = week if week is not None else None

    # Overview row for selectedWeek
    if selected_week is not None and "week" in pdf.columns:
        row_df = pdf[pdf["week"] == selected_week]
        overview = row_df.iloc[0].to_dict() if not row_df.empty else None
    else:
        overview = None

    # Weekly log sorted by week
    if "week" in pdf.columns:
        pdf_sorted = pdf.sort_values("week")
    else:
        pdf_sorted = pdf

    weekly_log = pdf_sorted.to_dict(orient="records")

    # Identity fields (from the latest row)
    first = pdf_sorted.iloc[-1]
    name = first.get("player_display_name") or first.get("player_name")
    team = first.get("team")
    position = first.get("position")

    identity = {
        "id": player_id,
        "name": name,
        "team": team,
        "position": position,
    }

    overview_clean = clean_nans(overview)
    weekly_log_clean = clean_nans(weekly_log)

    return {
        "ok": True,
        "player": identity,
        "selectedWeek": selected_week,
        "weeksWithData": weeks_with_data,
        "overview": overview_clean,
        "weeklyLog": weekly_log_clean,
    }


# -------------------------------------------------------
# AI picks: yards / receptions / PPR
# -------------------------------------------------------

VALID_STATS: Dict[str, str] = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "fantasy_points_ppr": "fantasy_points_ppr",
}


@app.get("/ai_picks")
def ai_picks(
    week: Optional[int] = None,
    position: Optional[str] = None,
    stat: str = "passing_yards",
    limit: int = 20,
):
    """
    Matchup-aware projection endpoint.

    For a target week:
      - Looks at all games BEFORE that week
      - Projects the chosen stat for each offensive player
      - Adjusts based on:
          * opponent defense vs that stat (rank & softness)
          * home vs away
          * rough prime-time flag
      - Returns projection + confidence + matchup context

    Default behavior:
      - If no week is given, target_week = current data week (max_week)
      - History uses weeks < target_week (so for week 13, uses weeks 1–12)
    """
    df = load_weekly_stats()

    if "week" not in df.columns:
        return {
            "ok": False,
            "error": "No 'week' column in data",
            "columns": list(df.columns),
        }

    if stat not in VALID_STATS:
        return {
            "ok": False,
            "error": (
                f"Unsupported stat '{stat}'. "
                f"Valid options: {sorted(VALID_STATS.keys())}"
            ),
        }

    stat_col = VALID_STATS[stat]

    max_week = int(df["week"].max())

    # ✅ Default to CURRENT week, not next week
    if week is None:
        target_week = max_week
    else:
        target_week = int(week)

    # Historical games only (no peeking into the target week)
    hist = df[df["week"] < target_week].copy()

    # Positions to consider
    if position is not None:
        hist = hist[hist["position"] == position]
    else:
        hist = hist[hist["position"].isin(["QB", "RB", "WR", "TE"])]

    if hist.empty:
        return {
            "ok": False,
            "error": f"No historical data found before week {target_week} "
            f"for the given filters.",
        }

    if stat_col not in hist.columns:
        return {
            "ok": False,
            "error": f"Stat column '{stat_col}' not found in data.",
            "columns": list(hist.columns),
        }

    # ---------------------------
    # Defense strength vs stat
    # ---------------------------
    # For each defense (opponent_team), how much of this stat do they allow per game?
    def_allowed = (
        hist.groupby("opponent_team")[stat_col]
        .mean()
        .dropna()
        .rename("def_allowed_avg")
    )

    def_df = def_allowed.reset_index().rename(columns={"opponent_team": "def_team"})

    if not def_df.empty:
        # Rank defenses: 1 = toughest (lowest allowed), N = softest (highest allowed)
        def_df["defense_rank_vs_stat"] = def_df["def_allowed_avg"].rank(
            method="min", ascending=True
        ).astype(int)
        max_rank = int(def_df["defense_rank_vs_stat"].max())
        # Softness: 0 = toughest, 1 = softest
        def_df["defense_softness"] = (
            (def_df["defense_rank_vs_stat"] - 1) / max(1, (max_rank - 1))
        )
        defense_map = def_df.set_index("def_team")[
            ["def_allowed_avg", "defense_rank_vs_stat", "defense_softness"]
        ]
    else:
        defense_map = None

    # ---------------------------
    # Schedule for target week:
    # home/away + prime-time
    # ---------------------------
    schedule_df = load_schedule()
    schedule_map = None
    if schedule_df is not None and {
        "season",
        "week",
        "home_team",
        "away_team",
    }.issubset(schedule_df.columns):
        sched = schedule_df[
            (schedule_df["season"] == SEASON)
            & (schedule_df["week"] == target_week)
        ].copy()

        records: List[Dict[str, Any]] = []
        for _, row in sched.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            gameday = row.get("gameday")
            gametime = row.get("gametime")
            weekday = row.get("weekday")

            # Rough prime-time flag: MNF/TNF/SNF or time >= 20:00
            is_night = False
            try:
                if isinstance(gametime, str) and len(gametime) >= 2:
                    hour = int(gametime.split(":")[0])
                    if hour >= 20:
                        is_night = True
                if isinstance(weekday, str) and weekday.upper() in {
                    "MONDAY",
                    "THURSDAY",
                    "SUNDAY",
                }:
                    is_night = True or is_night
            except Exception:
                pass

            # Home team view
            records.append(
                {
                    "team": home,
                    "opponent": away,
                    "is_home": True,
                    "is_prime_time": bool(is_night),
                    "gameday": gameday,
                    "gametime": gametime,
                    "weekday": weekday,
                }
            )
            # Away team view
            records.append(
                {
                    "team": away,
                    "opponent": home,
                    "is_home": False,
                    "is_prime_time": bool(is_night),
                    "gameday": gameday,
                    "gametime": gametime,
                    "weekday": weekday,
                }
            )

        if records:
            sched_map_df = pd.DataFrame.from_records(records)
            schedule_map = sched_map_df.set_index("team")

    # ---------------------------
    # Build picks
    # ---------------------------
    picks: List[Dict[str, Any]] = []

    grouped = hist.groupby("player_id")

    for player_id, g in grouped:
        stat_series = g[stat_col].dropna()
        games_played = int((~stat_series.isna()).sum())
        if games_played < 3:
            # Need a little history to say anything reasonable
            continue

        avg = float(stat_series.mean())
        std = float(stat_series.std(ddof=0)) if games_played > 1 else 0.0
        last3_avg = float(stat_series.tail(3).mean())

        # Base projection from averages
        projection = 0.6 * avg + 0.4 * last3_avg

        # Consistency: 0–1, higher = more consistent
        epsilon = 1e-6
        variability = std / (avg + epsilon) if avg > 0 else 1.0
        consistency = 1.0 / (1.0 + variability)

        # Base confidence from sample size (0–1)
        base_conf = min(1.0, games_played / 10.0)

        # Identity fields
        last_row = g.iloc[-1]
        name = last_row.get("player_display_name") or last_row.get("player_name")
        team = last_row.get("team")
        pos = last_row.get("position")

        # Matchup context for target week
        opponent = None
        is_home = None
        is_prime = None
        gameday = None
        gametime = None
        weekday = None

        if schedule_map is not None and team in schedule_map.index:
            match = schedule_map.loc[team]
            opponent = match.get("opponent")
            is_home = bool(match.get("is_home"))
            is_prime = bool(match.get("is_prime_time"))
            gameday = match.get("gameday")
            gametime = match.get("gametime")
            weekday = match.get("weekday")

        # Defense vs this stat
        def_rank = None
        def_allowed_avg = None
        defense_softness = None

        if opponent and defense_map is not None and opponent in defense_map.index:
            drow = defense_map.loc[opponent]
            def_allowed_avg = float(drow["def_allowed_avg"])
            def_rank = int(drow["defense_rank_vs_stat"])
            defense_softness = float(drow["defense_softness"])

        # Apply context multipliers
        context_mult_conf = 1.0
        context_mult_proj = 1.0

        # Defense softness: 0 = toughest, 1 = softest
        if defense_softness is not None:
            # Boost a little vs softer defenses, small penalty vs tough
            context_mult_conf *= 0.9 + 0.2 * defense_softness  # ~0.9–1.1
            context_mult_proj *= 0.95 + 0.1 * defense_softness  # ~0.95–1.05

        # Home field advantage
        if is_home is True:
            context_mult_conf *= 1.05
            context_mult_proj *= 1.03
        elif is_home is False:
            context_mult_conf *= 0.97
            context_mult_proj *= 0.98

        # Prime time – tiny bump
        if is_prime is True:
            context_mult_conf *= 1.02

        # Final projection / confidence
        projection_adj = projection * context_mult_proj
        confidence_raw = 100 * base_conf * consistency * context_mult_conf
        confidence = max(1, min(100, int(round(confidence_raw))))
        rating = max(1, min(5, confidence // 20))

        pick = {
            "player_id": player_id,
            "name": name,
            "team": team,
            "position": pos,
            "stat": stat,
            "target_week": target_week,
            "projection": projection_adj,
            "games_sampled": games_played,
            "avg": avg,
            "last3_avg": last3_avg,
            "std": std,
            "consistency": consistency,
            "confidence": confidence,  # 1–100
            "rating": rating,          # 1–5

            # Matchup info
            "opponent": opponent,
            "defense_allowed_avg": def_allowed_avg,
            "defense_rank_vs_stat": def_rank,
            "is_home": is_home,
            "is_prime_time": is_prime,
            "gameday": gameday,
            "gametime": gametime,
            "weekday": weekday,
        }

        picks.append(pick)

    if not picks:
        return {
            "ok": False,
            "error": "No players had enough history for this stat.",
        }

    # Score: projection * (confidence scaled) – simple ranking
    def pick_score(p: Dict[str, Any]) -> float:
        return float(p["projection"]) * (p["confidence"] / 100.0)

    picks_sorted = sorted(picks, key=pick_score, reverse=True)
    picks_top = picks_sorted[: max(1, int(limit))]

    picks_clean = clean_nans(picks_top)

    return {
        "ok": True,
        "week": target_week,
        "stat": stat,
        "position": position,
        "count": len(picks_clean),
        "picks": picks_clean,
    }


# -------------------------------------------------------
# AI TD picks: "anytime TD" style signal
# -------------------------------------------------------


@app.get("/ai_td_picks")
def ai_td_picks(
    week: Optional[int] = None,
    position: Optional[str] = None,
    limit: int = 20,
):
    """
    Crude "anytime TD" style projection.

    For a target week:
      - Looks at all games BEFORE that week
      - Computes TD rate from passing/rushing/receiving TDs
      - Adjusts with same matchup/home/primetime logic as /ai_picks
      - Outputs a TD "projection" and confidence (0–100)

    This is still just analytics/education, not betting advice.
    """
    df = load_weekly_stats()

    if "week" not in df.columns:
        return {
            "ok": False,
            "error": "No 'week' column in data",
            "columns": list(df.columns),
        }

    required_td_cols = {"passing_tds", "rushing_tds", "receiving_tds"}
    if not required_td_cols.issubset(df.columns):
        return {
            "ok": False,
            "error": "Missing TD columns in data",
            "missing": sorted(required_td_cols - set(df.columns)),
            "columns": list(df.columns),
        }

    max_week = int(df["week"].max())

    # ✅ Default to CURRENT week, not next
    if week is None:
        target_week = max_week
    else:
        target_week = int(week)

    # Historical games only
    hist = df[df["week"] < target_week].copy()

    # Positions to consider
    if position is not None:
        hist = hist[hist["position"] == position]
    else:
        hist = hist[hist["position"].isin(["QB", "RB", "WR", "TE"])]

    if hist.empty:
        return {
            "ok": False,
            "error": f"No historical data found before week {target_week} "
            f"for the given filters.",
        }

    # Create a simple "td_events" per game
    hist["td_events"] = (
        hist["passing_tds"].fillna(0)
        + hist["rushing_tds"].fillna(0)
        + hist["receiving_tds"].fillna(0)
    )

    # Defense vs TDs: how many TD events they allow per game
    def_td_allowed = (
        hist.groupby("opponent_team")["td_events"]
        .mean()
        .dropna()
        .rename("def_td_allowed_avg")
    )

    def_td_df = def_td_allowed.reset_index().rename(
        columns={"opponent_team": "def_team"}
    )

    if not def_td_df.empty:
        def_td_df["defense_rank_vs_td"] = def_td_df["def_td_allowed_avg"].rank(
            method="min", ascending=True
        ).astype(int)
        max_rank_td = int(def_td_df["defense_rank_vs_td"].max())
        def_td_df["defense_td_softness"] = (
            (def_td_df["defense_rank_vs_td"] - 1) / max(1, (max_rank_td - 1))
        )
        defense_td_map = def_td_df.set_index("def_team")[
            ["def_td_allowed_avg", "defense_rank_vs_td", "defense_td_softness"]
        ]
    else:
        defense_td_map = None

    # Schedule for target week
    schedule_df = load_schedule()
    schedule_map = None
    if schedule_df is not None and {
        "season",
        "week",
        "home_team",
        "away_team",
    }.issubset(schedule_df.columns):
        sched = schedule_df[
            (schedule_df["season"] == SEASON)
            & (schedule_df["week"] == target_week)
        ].copy()

        records: List[Dict[str, Any]] = []
        for _, row in sched.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            gameday = row.get("gameday")
            gametime = row.get("gametime")
            weekday = row.get("weekday")

            is_night = False
            try:
                if isinstance(gametime, str) and len(gametime) >= 2:
                    hour = int(gametime.split(":")[0])
                    if hour >= 20:
                        is_night = True
                if isinstance(weekday, str) and weekday.upper() in {
                    "MONDAY",
                    "THURSDAY",
                    "SUNDAY",
                }:
                    is_night = True or is_night
            except Exception:
                pass

            records.append(
                {
                    "team": home,
                    "opponent": away,
                    "is_home": True,
                    "is_prime_time": bool(is_night),
                    "gameday": gameday,
                    "gametime": gametime,
                    "weekday": weekday,
                }
            )
            records.append(
                {
                    "team": away,
                    "opponent": home,
                    "is_home": False,
                    "is_prime_time": bool(is_night),
                    "gameday": gameday,
                    "gametime": gametime,
                    "weekday": weekday,
                }
            )

        if records:
            sched_map_df = pd.DataFrame.from_records(records)
            schedule_map = sched_map_df.set_index("team")

    picks: List[Dict[str, Any]] = []

    grouped = hist.groupby("player_id")

    for player_id, g in grouped:
        td_series = g["td_events"].dropna()
        games_played = int((~td_series.isna()).sum())
        if games_played < 3:
            continue

        avg_td = float(td_series.mean())
        std_td = float(td_series.std(ddof=0)) if games_played > 1 else 0.0
        last3_td = float(td_series.tail(3).mean())

        base_proj = 0.6 * avg_td + 0.4 * last3_td

        epsilon = 1e-6
        variability = std_td / (avg_td + epsilon) if avg_td > 0 else 1.0
        consistency = 1.0 / (1.0 + variability)

        base_conf = min(1.0, games_played / 10.0)

        last_row = g.iloc[-1]
        name = last_row.get("player_display_name") or last_row.get("player_name")
        team = last_row.get("team")
        pos = last_row.get("position")

        opponent = None
        is_home = None
        is_prime = None
        gameday = None
        gametime = None
        weekday = None

        if schedule_map is not None and team in schedule_map.index:
            match = schedule_map.loc[team]
            opponent = match.get("opponent")
            is_home = bool(match.get("is_home"))
            is_prime = bool(match.get("is_prime_time"))
            gameday = match.get("gameday")
            gametime = match.get("gametime")
            weekday = match.get("weekday")

        def_rank_td = None
        def_td_allowed_avg = None
        defense_td_softness = None

        if opponent and defense_td_map is not None and opponent in defense_td_map.index:
            drow = defense_td_map.loc[opponent]
            def_td_allowed_avg = float(drow["def_td_allowed_avg"])
            def_rank_td = int(drow["defense_rank_vs_td"])
            defense_td_softness = float(drow["defense_td_softness"])

        context_mult_conf = 1.0
        context_mult_proj = 1.0

        if defense_td_softness is not None:
            context_mult_conf *= 0.9 + 0.2 * defense_td_softness
            context_mult_proj *= 0.95 + 0.1 * defense_td_softness

        if is_home is True:
            context_mult_conf *= 1.05
            context_mult_proj *= 1.03
        elif is_home is False:
            context_mult_conf *= 0.97
            context_mult_proj *= 0.98

        if is_prime is True:
            context_mult_conf *= 1.02

        proj_td_adj = base_proj * context_mult_proj
        confidence_raw = 100 * base_conf * consistency * context_mult_conf
        confidence = max(1, min(100, int(round(confidence_raw))))
        rating = max(1, min(5, confidence // 20))

        pick = {
            "player_id": player_id,
            "name": name,
            "team": team,
            "position": pos,
            "target_week": target_week,
            "td_projection": proj_td_adj,
            "games_sampled": games_played,
            "avg_td": avg_td,
            "last3_td": last3_td,
            "std_td": std_td,
            "consistency": consistency,
            "confidence": confidence,
            "rating": rating,
            "opponent": opponent,
            "def_td_allowed_avg": def_td_allowed_avg,
            "defense_rank_vs_td": def_rank_td,
            "is_home": is_home,
            "is_prime_time": is_prime,
            "gameday": gameday,
            "gametime": gametime,
            "weekday": weekday,
        }

        picks.append(pick)

    if not picks:
        return {
            "ok": False,
            "error": "No players had enough TD history.",
        }

    def pick_score(p: Dict[str, Any]) -> float:
        return float(p["td_projection"]) * (p["confidence"] / 100.0)

    picks_sorted = sorted(picks, key=pick_score, reverse=True)
    picks_top = picks_sorted[: max(1, int(limit))]

    picks_clean = clean_nans(picks_top)

    return {
        "ok": True,
        "week": target_week,
        "position": position,
        "count": len(picks_clean),
        "picks": picks_clean,
    }


# -------------------------------------------------------
# Local dev entrypoint
# -------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
