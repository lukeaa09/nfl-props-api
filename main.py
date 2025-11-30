from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import nflreadpy as nfl
import pandas as pd
import polars as pl
from typing import Any, Optional

SEASON = 2025

app = FastAPI(
    title="NFL Props API",
    description="Simple API that exposes nflverse weekly player stats and basic AI picks",
    version="3.1.0",
)

# Allow your frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down later to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Global error handler
# -----------------------
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": repr(exc),
        },
    )


# -----------------------
# Utility: clean NaNs
# -----------------------
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


# -----------------------
# Load weekly stats
# -----------------------
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

    # Polars -> pandas
    if isinstance(pl_df, pl.DataFrame):
        df = pl_df.to_pandas()
    elif isinstance(pl_df, pd.DataFrame):
        df = pl_df
    else:
        df = pd.DataFrame(pl_df)

    return df


# -----------------------
# Load schedule (for home/away, opponent, primetime)
# -----------------------
def load_schedule() -> Optional[pd.DataFrame]:
    """
    Load NFL schedule for this season.

    Used to figure out:
      - who each team plays in a given week
      - whether they're home or away
      - whether it's (roughly) a night game
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


# -----------------------
# Basic endpoints
# -----------------------
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
    Just for debugging: shows columns + a few sample rows.
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


# -----------------------
# /players: list/search
# -----------------------
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


# -----------------------
# /player/{id}: details
# -----------------------
@app.get("/player/{player_id}")
def player_detail(player_id: str, week: Optional[int] = None):
    """
    Return weekly log + overview for a single player.
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

    # Identity fields
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


# -----------------------
# AI picks (matchup-aware)
# -----------------------
VALID_STATS = {
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
    Matchup-aware 'AI picks' endpoint.

    For a target week:
      - Looks at all games BEFORE that week
      - Projects the stat for each offensive player
      - Adjusts based on:
          * opponent defense vs that stat (rank & softness)
          * home vs away
          * rough primetime flag
      - Returns projection + confidence + matchup context

    This is for analytics / education only, not betting advice.
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

    # Target week: default = next week after last completed week
    if week is None:
        target_week = max_week
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
        # Softness score: 0 = toughest, 1 = softest
        def_df["defense_softness"] = (
            (def_df["defense_rank_vs_stat"] - 1) / max(1, (max_rank - 1))
        )
        defense_map = def_df.set_index("def_team")[
            ["def_allowed_avg", "defense_rank_vs_stat", "defense_softness"]
        ]
    else:
        defense_map = None

    # ---------------------------
    # Schedule: home/away + primetime for target week
    # ---------------------------
    schedule_df = load_schedule()
    schedule_map = None
    if schedule_df is not None and {"season", "week", "home_team", "away_team"}.issubset(
        schedule_df.columns
    ):
        sched = schedule_df[
            (schedule_df["season"] == SEASON) & (schedule_df["week"] == target_week)
        ].copy()

        records = []
        for _, row in sched.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            gameday = row.get("gameday")
            gametime = row.get("gametime")
            weekday = row.get("weekday")

            # Very rough primetime flag: MNF/TNF/SNF or time >= 20:00
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

        # Primetime – tiny bump
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

    # Score: projection * (confidence scaled) – simple ranking
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


# -----------------------
# AI TD picks (anytime TD style)
# -----------------------
@app.get("/ai_td_picks")
def ai_td_picks(
    week: Optional[int] = None,
    position: Optional[str] = None,
    limit: int = 20,
):
    """
    Simple "anytime TD" style projections.

    Uses:
      td_events = passing_tds + rushing_tds + receiving_tds

    For each offensive player:
      - looks at games before target week
      - computes average & last3 avg TDs
      - builds a projection + confidence

    Educational/analytics only, not betting advice.
    """
    df = load_weekly_stats()

    required_cols = {"player_id", "player_display_name", "team", "position",
                     "week", "passing_tds", "rushing_tds", "receiving_tds"}
    missing = required_cols - set(df.columns)
    if missing:
        return {
            "ok": False,
            "error": f"Missing required columns for TD picks: {sorted(missing)}",
            "columns": list(df.columns),
        }

    max_week = int(df["week"].max())
    if week is None:
        target_week = max_week + 1
    else:
        target_week = int(week)

    hist = df[df["week"] < target_week].copy()

    # Offensive positions only
    if position is not None:
        hist = hist[hist["position"] == position]
    else:
        hist = hist[hist["position"].isin(["QB", "RB", "WR", "TE"])]

    if hist.empty:
        return {
            "ok": False,
            "error": f"No historical data found before week {target_week} for the given filters.",
        }

    # TD events
    hist["td_events"] = (
        hist["passing_tds"].fillna(0)
        + hist["rushing_tds"].fillna(0)
        + hist["receiving_tds"].fillna(0)
    )

    picks: list[dict] = []
    grouped = hist.groupby("player_id")

    for player_id, g in grouped:
        td_series = g["td_events"].fillna(0)
        games_played = int(len(td_series))
        if games_played < 3:
            continue

        avg_td = float(td_series.mean())
        last3_avg_td = float(td_series.tail(3).mean())
        std_td = float(td_series.std(ddof=0)) if games_played > 1 else 0.0

        # If they basically never score, skip as a TD prop candidate
        if avg_td < 0.1 and last3_avg_td < 0.1:
            continue

        # Base projection
        td_projection = 0.6 * avg_td + 0.4 * last3_avg_td

        # Consistency & confidence
        epsilon = 1e-6
        variability = std_td / (avg_td + epsilon) if avg_td > 0 else 1.0
        consistency = 1.0 / (1.0 + variability)
        base_conf = min(1.0, games_played / 10.0)

        confidence_raw = 100 * base_conf * consistency
        confidence = max(1, min(100, int(round(confidence_raw))))
        rating = max(1, min(5, confidence // 20))

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
                "target_week": target_week,
                "td_projection": td_projection,
                "games_sampled": games_played,
                "avg_tds": avg_td,
                "last3_avg_tds": last3_avg_td,
                "std_tds": std_td,
                "consistency": consistency,
                "confidence": confidence,
                "rating": rating,
            }
        )

    if not picks:
        return {
            "ok": False,
            "error": "No players had enough TD history to project.",
        }

    def score(p: dict) -> float:
        return float(p["td_projection"]) * (p["confidence"] / 100.0)

    picks_sorted = sorted(picks, key=score, reverse=True)
    top = picks_sorted[: max(1, int(limit))]
    top_clean = clean_nans(top)

    return {
        "ok": True,
        "week": target_week,
        "position": position,
        "count": len(top_clean),
        "picks": top_clean,
    }


# -----------------------
# Combined AI picks across all stats
# -----------------------
COMBINED_STATS_FOR_PICKS = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "fantasy_points_ppr",
]


@app.get("/ai_picks_combined")
def ai_picks_combined(
    week: Optional[int] = None,
    limit: int = 10,
):
    """
    Combined AI picks across multiple stats.

    - Calls ai_picks() for each stat in COMBINED_STATS_FOR_PICKS
    - Merges all picks into one list
    - Keeps the highest-confidence pick per (player_id, stat)
    - Sorts by confidence desc, then projection desc
    - Returns the top `limit` picks
    """

    all_picks: list[dict] = []
    result_week: Optional[int] = None

    for stat in COMBINED_STATS_FOR_PICKS:
        resp = ai_picks(
            week=week,
            position=None,
            stat=stat,
            limit=100,
        )

        if not isinstance(resp, dict) or not resp.get("ok"):
            continue

        if result_week is None:
            result_week = resp.get("week")

        picks = resp.get("picks", [])
        if isinstance(picks, list):
            all_picks.extend(picks)

    if not all_picks:
        return {
            "ok": False,
            "error": "No picks were available from underlying stats.",
        }

    # Deduplicate by (player_id, stat) and keep the highest-confidence version
    unique: dict[tuple[str, str | None], dict] = {}
    for p in all_picks:
        player_id = p.get("player_id")
        stat = p.get("stat")
        if not player_id:
            continue

        key = (str(player_id), str(stat) if stat is not None else None)
        existing = unique.get(key)
        if existing is None or p.get("confidence", 0) > existing.get("confidence", 0):
            unique[key] = p

    combined_list = list(unique.values())

    # Sort by confidence (desc), then projection (desc)
    def sort_key(p: dict) -> tuple[float, float]:
        conf = float(p.get("confidence", 0))
        proj = float(p.get("projection", 0.0))
        return (conf, proj)

    combined_list.sort(key=sort_key, reverse=True)

    top_n = combined_list[: max(1, int(limit))]
    top_clean = clean_nans(top_n)

    return {
        "ok": True,
        "week": result_week,
        "limit": int(limit),
        "count": len(top_clean),
        "picks": top_clean,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
