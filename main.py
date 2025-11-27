import io
import os
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# CONFIG
# -------------------------------

SEASON = 2025

# Official nflverse player stats parquet (all seasons)
PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/"
    "releases/download/player_stats/player_stats.parquet"
)

# The Odds API key (you already set this in Render)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")


# -------------------------------
# DATA LOADERS
# -------------------------------

def load_weekly_stats() -> pd.DataFrame:
    """
    Load weekly player stats for SEASON from nflverse player_stats.parquet.

    Filters to:
      - selected SEASON
      - regular season only
      - offensive skill positions (QB/RB/WR/TE)
    Cleans numeric columns so JSON doesn't blow up on NaN/inf.
    """
    try:
        print("[startup] Downloading player_stats.parquet ...")
        resp = requests.get(PLAYER_STATS_URL, timeout=90)
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)
        df = pd.read_parquet(buf)
    except Exception as e:
        print("ERROR loading player_stats parquet:", e)
        return pd.DataFrame()

    # Filter to this season + regular season
    if "season" in df.columns:
        df = df[df["season"] == SEASON]

    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]

    # Filter to offensive players only
    if "position_group" in df.columns:
        df = df[df["position_group"].isin(["QB", "RB", "WR", "TE"])]
    elif "position" in df.columns:
        df = df[df["position"].isin(["QB", "RB", "WR", "TE"])]

    # Drop rows without player_id or week
    if "player_id" not in df.columns or "week" not in df.columns:
        print("WARNING: player_stats parquet missing player_id or week columns")
        return pd.DataFrame()

    df = df[df["player_id"].notna() & df["week"].notna()]

    # Normalize types
    df["player_id"] = df["player_id"].astype(str)
    if "player_display_name" in df.columns:
        df["player_display_name"] = df["player_display_name"].fillna("Unknown Player")
    else:
        if "player_name" in df.columns:
            df["player_display_name"] = df["player_name"].fillna("Unknown Player")
        else:
            df["player_display_name"] = "Unknown Player"

    # Clean numeric columns
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


def build_players_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact index of players for search:
      - player_id
      - name
      - team
      - position
    """
    if df.empty:
        return pd.DataFrame(columns=["player_id", "name", "team", "position", "name_lower"])

    latest = (
        df.sort_values(["player_id", "week"])
          .groupby("player_id")
          .tail(1)
    )

    cols = []
    for candidate in ["player_display_name", "player_name"]:
        if candidate in latest.columns:
            cols.append(candidate)
            break

    cols.extend([c for c in ["team", "position"] if c in latest.columns])

    base = latest[["player_id"] + cols].drop_duplicates()

    if "player_display_name" in base.columns:
        base = base.rename(columns={"player_display_name": "name"})
    elif "player_name" in base.columns:
        base = base.rename(columns={"player_name": "name"})
    else:
        base["name"] = "Unknown Player"

    if "team" not in base.columns:
        base["team"] = ""
    if "position" not in base.columns:
        base["position"] = ""

    base["name_lower"] = base["name"].str.lower()

    return base[["player_id", "name", "team", "position", "name_lower"]]


def sanitize_row(row: pd.Series) -> Dict[str, Any]:
    """
    Convert a pandas row to a JSON-safe dict (no NaN/inf).
    """
    out: Dict[str, Any] = {}
    as_dict = row.to_dict()
    for k, v in as_dict.items():
        if isinstance(v, float):
            if pd.isna(v) or v == float("inf") or v == float("-inf"):
                out[k] = None
            else:
                out[k] = float(v)
        else:
            out[k] = v
    return out


def sanitize_frame(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [sanitize_row(r) for _, r in df.iterrows()]


def get_player_payload(player_id: str, week: Optional[int] = None) -> Dict[str, Any]:
    """
    Core logic used by both:
      - /players/{player_id}
      - /players/{player_id}/week/{week}
    """
    if WEEKLY.empty:
        raise HTTPException(status_code=503, detail="Stats not loaded")

    player_df = WEEKLY[WEEKLY["player_id"] == player_id]
    if player_df.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    weeks_with_data = sorted(player_df["week"].astype(int).unique().tolist())

    if week is None:
        selected_week = max(weeks_with_data)
    else:
        if week not in weeks_with_data:
            raise HTTPException(status_code=404, detail="No data for that week")
        selected_week = week

    overview_row = player_df[player_df["week"] == selected_week].iloc[0]
    overview = sanitize_row(overview_row)

    weekly_log = sanitize_frame(player_df.sort_values("week"))

    player_info = {
        "id": overview.get("player_id"),
        "name": overview.get("player_display_name"),
        "team": overview.get("team"),
        "position": overview.get("position"),
    }

    return {
        "ok": True,
        "player": player_info,
        "selectedWeek": int(selected_week),
        "weeksWithData": weeks_with_data,
        "overview": overview,
        "weeklyLog": weekly_log,
    }


# -------------------------------
# LOAD DATA AT STARTUP
# -------------------------------

WEEKLY: pd.DataFrame = load_weekly_stats()
PLAYERS: pd.DataFrame = build_players_index(WEEKLY)

print(f"[startup] Loaded {len(WEEKLY)} weekly rows for season {SEASON}")
print(f"[startup] Indexed {len(PLAYERS)} players")


# -------------------------------
# FASTAPI APP
# -------------------------------

app = FastAPI(title="NFL Props API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# ROOT / HEALTH
# -------------------------------

@app.get("/")
def root():
    return {
        "ok": True,
        "season": SEASON,
        "players_loaded": int(PLAYERS.shape[0]),
        "weekly_rows": int(WEEKLY.shape[0]),
        "has_odds_api_key": bool(ODDS_API_KEY),
    }


# -------------------------------
# PLAYER SEARCH
# -------------------------------

def _search_players(q: str) -> Dict[str, Any]:
    if PLAYERS.empty:
        return {"players": []}

    q_lower = q.strip().lower()
    if not q_lower:
        result = PLAYERS.sort_values("name").head(50)
    else:
        mask = PLAYERS["name_lower"].str.contains(q_lower)
        result = PLAYERS[mask].sort_values("name").head(50)

    players = [
        {
            "id": row["player_id"],
            "name": row["name"],
            "team": row["team"],
            "position": row["position"],
        }
        for _, row in result.iterrows()
    ]
    return {"players": players}


@app.get("/players")
def players(q: str = Query("", description="Search by player name")):
    """
    Example:
      /players?q=mahomes
    """
    return _search_players(q)


@app.get("/players/search")
def players_search(q: str = Query("", description="Search by player name")):
    """
    Same as /players for compatibility.
    """
    return _search_players(q)


# -------------------------------
# PLAYER DETAILS
# -------------------------------

@app.get("/players/{player_id}")
def player_details(player_id: str, week: Optional[int] = Query(None)):
    """
    Example:
      /players/00-0033873           -> latest week
      /players/00-0033873?week=8    -> specific week
    """
    return get_player_payload(player_id, week)


@app.get("/players/{player_id}/week/{week}")
def player_details_week(player_id: str, week: int):
    """
    Example:
      /players/00-0033873/week/8
    """
    return get_player_payload(player_id, week)


# -------------------------------
# ODDS API PASSTHROUGH
# -------------------------------

@app.get("/player-props/week/{week}")
def player_props_week(week: int):
    """
    Raw passthrough from The Odds API for now.
    We'll hook this into your picks logic later.
    """
    if not ODDS_API_KEY:
        raise HTTPException(status_code=500, detail="ODDS_API_KEY not set")

    url = (
        "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        f"?apiKey={ODDS_API_KEY}"
        "&regions=us"
        "&markets=player_pass_yds,player_rush_yds,player_rec_yds"
        "&oddsFormat=american"
    )

    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Odds API error: {resp.text}",
            )
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to call Odds API: {e}")
