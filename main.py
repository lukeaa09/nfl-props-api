from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import nflreadpy as nfl
import pandas as pd
import polars as pl
from typing import Any

SEASON = 2025

app = FastAPI(
    title="NFL Props API",
    description="Simple API that exposes nflverse weekly player stats",
    version="2.2.0",
)

# Allow your frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 🔴 GLOBAL ERROR HANDLER: never return HTML 500, always JSON
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": repr(exc),
        },
    )


def clean_nans(value: Any) -> Any:
    """
    Recursively replace NaN/NaT with None so JSON encoding doesn't break.
    """
    if isinstance(value, float):
        if value != value:  # NaN check
            return None
        return value
    if isinstance(value, dict):
        return {k: clean_nans(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_nans(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def load_weekly_stats() -> pd.DataFrame:
    def load_schedule() -> pd.DataFrame | None:
    """
    Load NFL schedule for this season.

    Used to figure out:
      - who each team plays in a given week
      - whether they're home or away
      - whether it's a night game (rough approximation)
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

    """
    Load nflverse weekly player stats for one season.

    By removing the cache, this will always pull the latest data
    as nflverse updates (new weeks, updated stats, etc.).
    """
    pl_df = nfl.load_player_stats(
        seasons=[SEASON],
        summary_level="week",
    )

    # Polars -> pandas
    if isinstance(pl_df, pl.DataFrame):
        df = pl_df.to_pandas()
    elif isinstance(pl_df, pd.DataFrame):
        df = pl_df
    else:
        df = pd.DataFrame(pl_df)

    return df
    


@app.get("/health")
def health():
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
    Just for debugging: shows columns + 3 sample rows.
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


@app.get("/players")
def list_players(q: str | None = None):
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


@app.get("/player/{player_id}")
def player_detail(player_id: str, week: int | None = None):
    """
    Very defensive version of player detail:
    - Returns:
        ok: True/False
        player: { id, name, team, position }
        selectedWeek: int or None
        weeksWithData: [int, ...]
        overview: dict for the selected week, or None
        weeklyLog: list of dicts for all weeks for that player
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

    # figure out weeks with data
    if "week" in pdf.columns:
        weeks_with_data = sorted(int(w) for w in pdf["week"].unique())
    else:
        weeks_with_data = []

    # decide selected week
    if weeks_with_data:
        if week is None:
            selected_week = max(weeks_with_data)
        else:
            selected_week = int(week)
    else:
        selected_week = week if week is not None else None

    # build overview (single week)
    if selected_week is not None and "week" in pdf.columns:
        row_df = pdf[pdf["week"] == selected_week]
        overview = row_df.iloc[0].to_dict() if not row_df.empty else None
    else:
        overview = None

    # weekly log = all rows for this player (raw)
    if "week" in pdf.columns:
        pdf_sorted = pdf.sort_values("week")
    else:
        pdf_sorted = pdf
    weekly_log = pdf_sorted.to_dict(orient="records")

    # basic identity
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

    # 🔥 Clean NaNs before returning so JSON encoding doesn't break
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
from math import sqrt

# -----------------------
# Simple AI picks endpoint
# -----------------------

VALID_STATS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "fantasy_points_ppr": "fantasy_points_ppr",
}


VALID_STATS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "fantasy_points_ppr": "fantasy_points_ppr",
}


@app.get("/ai_picks")
def ai_picks(
    week: int | None = None,
    position: str | None = None,
    stat: str = "passing_yards",
    limit: int = 20,
):
    """
    Matchup-aware 'AI picks' endpoint.

    For a target week:
      - Looks at all games BEFORE that week
      - Projects the stat for each offensive player
      - Adjusts based on:
          * opponent defense vs that stat (rank)
          * home vs away
          * rough primetime flag
      - Returns projection + confidence + matchup context
    """

    df = load_weekly_stats()

    if stat not in VALID_STATS:
        return {
            "ok": False,
            "error": f"Unsupported stat '{stat}'. Valid options: {sorted(VALID_STATS.keys())}",
        }

    stat_col = VALID_STATS[stat]

    if "week" not in df.columns:
        return {
            "ok": False,
            "error": "No 'week' column in data",
            "columns": list(df.columns),
        }

    max_week = int(df["week"].max())

    # Target week: by default, "next" week after last one we have
    if week is None:
        target_week = max_week + 1
    else:
        target_week = int(week)

    # Historical games only (no peeking into target week)
    hist = df[df["week"] < target_week].copy()

    # Positions to consider
    if position is not None:
        hist = hist[hist["position"] == position]
    else:
        hist = hist[hist["position"].isin(["QB", "RB", "WR", "TE"])]

    if hist.empty:
        return {
            "ok": False,
            "error": f"No historical data found before week {target_week} for the given filters.",
        }

    if stat_col not in hist.columns:
        return {
            "ok": False,
            "error": f"Stat column '{stat_col}' not found in data.",
            "columns": list(hist.columns),
        }

    # ---------------------------
    # Defensive strength vs stat
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
        # Softness score: 0 = toughest, 1 = softest
        def_df["defense_softness"] = (
            (def_df["defense_rank_vs_stat"] - 1) / max(1, (max_rank - 1))
        )
    else:
        max_rank = 0

    defense_map = (
        def_df.set_index("def_team")[["def_allowed_avg", "defense_rank_vs_stat", "defense_softness"]]
        if not def_df.empty
        else None
    )

    # ---------------------------
    # Schedule: home/away + primetime for target week
    # ---------------------------
    schedule_df = load_schedule()
    schedule_map = None
    if schedule_df is not None and {"season", "week", "home_team", "away_team"}.issubset(
        schedule_df.columns
    ):
        # Only this season & target week
        sched = schedule_df[
            (schedule_df["season"] == SEASON) & (schedule_df["week"] == target_week)
        ].copy()

        # Build a lookup: (team -> matchup info)
        records = []
        for _, row in sched.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            gameday = row.get("gameday")
            gametime = row.get("gametime")
            weekday = row.get("weekday")

            # Very rough primetime flag: MNF, TNF, SNF or time >= 20:00
            is_night = False
            try:
                if isinstance(gametime, str) and len(gametime) >= 2:
                    hour = int(gametime.split(":")[0])
                    if hour >= 20:
                        is_night = True
                if isinstance(weekday, str) and weekday.upper() in {"MONDAY", "THURSDAY", "SUNDAY"}:
                    # Many night games are on these days; we just keep the flag if already set
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

    picks: list[dict] = []

    grouped = hist.groupby("player_id")

    for player_id, g in grouped:
        stat_series = g[stat_col].dropna()
        games_played = int((~stat_series.isna()).sum())
        if games_played < 3:
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

        # ---------------------------
        # Matchup context for target week
        # ---------------------------
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

        # ---------------------------
        # Apply context multipliers
        # ---------------------------
        context_mult_conf = 1.0
        context_mult_proj = 1.0

        # Defense softness: 0 = toughest, 1 = softest
        if defense_softness is not None:
            # Boost a little vs softer defenses, small penalty vs tough
            # Range ~0.9–1.1
            context_mult_conf *= 0.9 + 0.2 * defense_softness
            context_mult_proj *= 0.95 + 0.1 * defense_softness

        # Home field advantage
        if is_home is True:
            context_mult_conf *= 1.05
            context_mult_proj *= 1.03
        elif is_home is False:
            context_mult_conf *= 0.97
            context_mult_proj *= 0.98

        # Primetime – tiny bump just to reflect "spotlight"
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
            "confidence": confidence,  # 0–100
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

    # Score: projection * (confidence scaled) – simple but works
    def pick_score(p: dict) -> float:
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


    max_week = int(df["week"].max())
    if week is None:
        target_week = max_week + 1
    else:
        target_week = int(week)

    # Use ONLY weeks before the target week (no peeking into the future)
    hist = df[df["week"] < target_week].copy()

    # Offensive positions only, unless overridden
    if position is not None:
        hist = hist[hist["position"] == position]
    else:
        hist = hist[hist["position"].isin(["QB", "RB", "WR", "TE"])]

    if hist.empty:
        return {
            "ok": False,
            "error": f"No historical data found before week {target_week} for the given filters.",
        }

    if stat_col not in hist.columns:
        return {
            "ok": False,
            "error": f"Stat column '{stat_col}' not found in data.",
            "columns": list(hist.columns),
        }

    # Group by player and compute basic stats
    picks: list[dict] = []
    grouped = hist.groupby("player_id")

    for player_id, g in grouped:
        # Clean stat values
        stat_series = g[stat_col].dropna()

        # Require at least 3 games with that stat to trust it even a little
        games_played = int((~stat_series.isna()).sum())
        if games_played < 3:
            continue

        avg = float(stat_series.mean())
        # Population std dev (ddof=0)
        std = float(stat_series.std(ddof=0)) if games_played > 1 else 0.0

        # Last 3 games average (if fewer than 3, it's just whatever we have)
        last3_avg = float(stat_series.tail(3).mean())

        # Simple projection: blend season-long average and last-3 average
        projection = 0.6 * avg + 0.4 * last3_avg

        # Consistency: higher if they are steady, lower if super volatile
        # Add a tiny epsilon so we never divide by zero
        epsilon = 1e-6
        # "Coefficient of variation" type measure
        variability = std / (avg + epsilon) if avg > 0 else 1.0
        # Map variability to [0,1] where lower variability = closer to 1
        consistency = 1.0 / (1.0 + variability)

        # Base confidence grows with games played, capped at 1.0
        base_conf = min(1.0, games_played / 10.0)

        # Final confidence (0–100)
        confidence = int(round(100 * base_conf * consistency))

        # Map confidence to a 1–5 rating
        rating = max(1, min(5, confidence // 20))

        # Grab identity fields from one row
        last_row = g.iloc[-1]
        name = last_row.get("player_display_name") or last_row.get("player_name")
        team = last_row.get("team")
        pos = last_row.get("position")

        picks.append(
            {
                "player_id": player_id,
                "name": name,
                "team": team,
                "position": pos,
                "stat": stat,
                "projection": projection,
                "games_sampled": games_played,
                "avg": avg,
                "last3_avg": last3_avg,
                "std": std,
                "consistency": consistency,
                "confidence": confidence,  # 0–100
                "rating": rating,          # 1–5
            }
        )

    if not picks:
        return {
            "ok": False,
            "error": "No players had enough history for this stat.",
        }

    # Sort by "score": projection * consistency * base_confidence
    # (Confidence is already in the pick dict) – we recompute a rough score here
    def pick_score(p: dict) -> float:
        return float(p["projection"]) * (p["confidence"] / 100.0)

    picks_sorted = sorted(picks, key=pick_score, reverse=True)
    picks_top = picks_sorted[: max(1, int(limit))]

    # Clean NaNs just in case
    picks_clean = clean_nans(picks_top)

    return {
        "ok": True,
        "week": target_week,
        "stat": stat,
        "position": position,
        "count": len(picks_clean),
        "picks": picks_clean,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
