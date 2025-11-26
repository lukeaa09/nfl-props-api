import os
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import requests
import nfl_data_py as nfl
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------
# CONFIG
# ---------------------------------------
SEASON = 2025

# Odds API key from Render env
ODDS_API_KEY = os.getenv("ODDS_API_KEY")


# ---------------------------------------
# DATA LOADERS
# ---------------------------------------

def load_weekly_stats() -> pd.DataFrame:
    """
    Load weekly player stats for SEASON using nfl_data_py.
    Filters to:
      - this SEASON
      - regular season only
      - offensive players (QB/RB/WR/TE)
    Cleans NaN/inf so JSON is safe.
    """
    try:
        df = nfl.import_weekly_data([SEASON])
    except Exception as e:
        print("ERROR loading weekly data via nfl_data_py:", e)
        return pd.DataFrame()

    # Regular season only
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]

    # Only offensive skill positions
    if "position" in df.columns:
        df = df[df["position"].isin(["QB", "RB", "WR", "TE"])]

    # Drop rows without player_id or week
    df = df[df["player_id"].notna() & df["week"].notna()]

    # Normalize types
    df["player_id"] = df["player_id"].astype(str)
    if "player_display_name" in df.columns:
        df["player_display_name"] = df["player_display_name"].fillna("Unknown Player")
    else:
        df["player_display_name"] = df.get("player_name", "Unknown Player")

    # Clean numeric columns so we don't explode JSON with NaN/inf
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


def build_players_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build index used for player search:
      - id
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

    index = latest[[
        "player_id",
        "player_display_name",
        "team",
        "position"
    ]].drop_duplicates()

    index = index.rename(columns={"player_display_name": "name"})
    index["name_lower"] = index["name"].str.lower()
    return index


def sanitize_row(row: pd.Series) -> Dict[str, Any]:
    """
    Convert a pandas row to a JSON-safe dict (no NaN/inf).
    """
    out: Dict[str, Any] = {}
    for k, v in row.to_dict().items():
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


# ---------------------------------------
# LOAD DATA AT STARTUP
# ---------------------------------------

WEEKLY: pd.DataFrame = load_weekly_stats()
PLAYERS: pd.DataFrame = build_players_index(WEEKLY)

print(f"[startup] Loaded {len(WEEKLY)} weekly rows for season {SEASON}")
print(f"[startup] Indexed {len(PLAYERS)} players")


# ---------------------------------------
# FASTAPI APP
# ---------------------------------------

app = FastAPI(title="NFL Props API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------
# HEALTH CHECK
# ---------------------------------------

@app.get("/")
def root():
    return {
        "ok": True,
        "season": SEASON,
        "players_loaded": int(PLAYERS.shape[0]),
        "weekly_rows": int(WEEKLY.shape[0]),
        "has_odds_api_key": bool(ODDS_API_KEY),
    }


# ---------------------------------------
# /players/search?q=
# ---------------------------------------

@app.get("/players/search")
def search_players(q: str = Query("", description="Search by player name")):
    """
    Example:
      /players/search?q=mahomes
    """
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


# ---------------------------------------
# /players/{player_id}/week/{week}
# ---------------------------------------

@app.get("/players/{player_id}/week/{week}")
def player_week(player_id: str, week: int):
    """
    Example:
      /players/00-0033873/week/8
    Returns:
      - player basic info
      - selected week overview
      - list of weeksWithData
      - full weeklyLog
    """
    if WEEKLY.empty:
        raise HTTPException(status_code=503, detail="Stats not loaded")

    player_df = WEEKLY[WEEKLY["player_id"] == player_id]
    if player_df.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    weeks_with_data = sorted(player_df["week"].astype(int).unique().tolist())
    if week not in weeks_with_data:
        raise HTTPException(status_code=404, detail="No data for that week")

    overview_row = player_df[player_df["week"] == week].iloc[0]
    overview = sanitize_row(overview_row)

    weekly_log = sanitize_frame(
        player_df.sort_values("week")
    )

    player_info = {
        "id": overview.get("player_id"),
        "name": overview.get("player_display_name"),
        "team": overview.get("team"),
        "position": overview.get("position"),
    }

    return {
        "ok": True,
        "player": player_info,
        "selectedWeek": int(week),
        "weeksWithData": weeks_with_data,
        "overview": overview,
        "weeklyLog": weekly_log,
    }


# ---------------------------------------
# BASIC ODDS API PASSTHROUGH (can upgrade later)
# ---------------------------------------

@app.get("/player-props/week/{week}")
def player_props_week(week: int):
    """
    Right now this just fetches current player prop odds from The Odds API.
    We ignore the 'week' for now because The Odds API is event/date based,
    not NFL-week based. Later we can map by kickoff date.
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
        # For now, just return raw odds API data so we can inspect it
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to call Odds API: {e}")
