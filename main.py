import os
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------
# CONFIG
# ---------------------------------------
SEASON = 2025

# nflverse player stats file (all seasons, all players)
PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/"
    "releases/download/player_stats/player_stats.parquet"
)

# The Odds API (you already set THE_ODDS_API_KEY in Render)
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")

# ---------------------------------------
# DATA LOADERS
# ---------------------------------------

def load_weekly_stats() -> pd.DataFrame:
    """
    Load weekly player stats for the configured season from nflverse.
    Filters to:
      - this SEASON
      - regular season only
      - offensive players (QB/RB/WR/TE)
    """
    try:
        df = pd.read_parquet(PLAYER_STATS_URL)
    except Exception as e:
        print("ERROR loading player_stats parquet:", e)
        return pd.DataFrame()

    # Filter to this season + regular season
    df = df[(df["season"] == SEASON) & (df["season_type"] == "REG")]

    # Keep only offensive skill positions, no OL / defense / ST
    if "position_group" in df.columns:
        df = df[df["position_group"].isin(["QB", "RB", "WR", "TE"])]

    # Drop rows without player_id or week
    df = df[df["player_id"].notna() & df["week"].notna()]

    # Ensure string IDs & display names
    df["player_id"] = df["player_id"].astype(str)
    if "player_display_name" in df.columns:
        df["player_display_name"] = df["player_display_name"].fillna("Unknown Player")
    else:
        df["player_display_name"] = df.get("player_name", "Unknown Player")

    # Replace NaNs in numeric columns with 0 so we don't blow up JSON later
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


def build_players_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple index of players:
      - id
      - name
      - team
      - position
    Used by /players search.
    """
    if df.empty:
        return pd.DataFrame(columns=["player_id", "name", "team", "position", "name_lower"])

    # Take the *latest* row per player (highest week) for basic info
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
    Convert a pandas row to a JSON-safe dict:
    - Replace NaN / inf with None
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

app = FastAPI(title="NFL Props API", version="0.2.0")

# Allow your Loveable frontend and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------
# BASIC HEALTH CHECK
# ---------------------------------------

@app.get("/")
def root():
    return {
        "ok": True,
        "season": SEASON,
        "players_loaded": int(PLAYERS.shape[0]),
        "weekly_rows": int(WEEKLY.shape[0]),
        "has_odds_api_key": bool(THE_ODDS_API_KEY),
    }


# ---------------------------------------
# PLAYERS SEARCH
# ---------------------------------------

@app.get("/players")
def search_players(q: str = Query("", description="Search by player name")):
    """
    Example:
      /players?q=mahomes
    """
    if PLAYERS.empty:
        return {"players": []}

    q_lower = q.strip().lower()
    if not q_lower:
        # Return a few popular players if no search term
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
# PLAYER DETAILS (WITH WEEKLY LOG)
# ---------------------------------------

@app.get("/players/{player_id}")
def player_details(player_id: str, week: Optional[int] = Query(None)):
    """
    Example:
      /players/00-0033873          -> full season log + latest week as overview
      /players/00-0033873?week=8   -> overview for that week + full season log
    """
    if WEEKLY.empty:
        raise HTTPException(status_code=503, detail="Stats not loaded")

    # All rows for this player
    player_df = WEEKLY[WEEKLY["player_id"] == player_id]
    if player_df.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    # List of weeks that have data
    weeks_with_data = sorted(player_df["week"].astype(int).unique().tolist())

    # Decide which week to highlight
    if week is not None and week in weeks_with_data:
        selected_week = week
    else:
        selected_week = max(weeks_with_data)

    # Row for selected week
    overview_row = player_df[player_df["week"] == selected_week]
    if overview_row.empty:
        raise HTTPException(status_code=404, detail="No data for selected week")

    overview = sanitize_row(overview_row.iloc[0])

    # Full weekly log
    weekly_log = sanitize_frame(
        player_df.sort_values("week")
    )

    # Basic player info from overview
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


# ---------------------------------------
# ODDS API RAW (we'll hook this into picks later)
# ---------------------------------------

@app.get("/odds/raw")
def odds_raw(
    regions: str = "us",
    markets: str = "player_pass_yds,player_rush_yds,player_rec_yds",
):
    """
    Simple passthrough to The Odds API so we can inspect data
    and later connect it to your picks page.

    Example:
      /odds/raw
    """
    if not THE_ODDS_API_KEY:
        raise HTTPException(status_code=500, detail="THE_ODDS_API_KEY not set")

    url = (
        "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        f"?apiKey={THE_ODDS_API_KEY}"
        f"&regions={regions}"
        f"&markets={markets}"
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
