from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
import pandas as pd
import nfl_data_py as nfl
import numpy as np
import requests
import os

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
SEASON = 2025
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")  # MUST BE SET IN RENDER ENV
ODDS_API_REGION = "us"
ODDS_API_SPORT = "americanfootball_nfl"

# -----------------------------------------------------------------------------
# APP SETUP
# -----------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
router_players = APIRouter()
router_stats = APIRouter()
router_odds = APIRouter()
router_ai = APIRouter()

# -----------------------------------------------------------------------------
# LOAD SCHEDULE
# -----------------------------------------------------------------------------
def load_schedule() -> pd.DataFrame | None:
    try:
        df = nfl.load_schedules(seasons=[SEASON])
        return df
    except Exception:
        return None

SCHEDULE = load_schedule()

def get_match_info(team: str, week: int):
    if SCHEDULE is None:
        return {"opponent": None, "home": None, "night": None}

    row = SCHEDULE[
        (SCHEDULE["team"] == team) &
        (SCHEDULE["week"] == week)
    ]

    if row.empty:
        return {"opponent": None, "home": None, "night": None}

    row = row.iloc[0]

    opponent = row.get("opponent", None)
    home = row.get("home", None)
    game_type = str(row.get("game_type", "")).lower()
    night = ("night" in game_type) or ("prime" in game_type)

    return {"opponent": opponent, "home": home, "night": night}

# -----------------------------------------------------------------------------
# LOAD WEEKLY PLAYER STATS
# -----------------------------------------------------------------------------
def load_weekly_stats():
    try:
        df = nfl.import_weekly_data([SEASON])
        df.replace([np.nan, np.inf, -np.inf], None, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

WEEKLY = load_weekly_stats()

# -----------------------------------------------------------------------------
# /players/search?q=
# -----------------------------------------------------------------------------
@router_players.get("/players/search")
def search_players(q: str):
    df = WEEKLY.copy()
    players = (
        df[["player_id", "player_display_name", "team", "position"]]
        .drop_duplicates()
    )

    mask = players["player_display_name"].str.contains(q, case=False, na=False)
    results = players[mask].head(20)

    return {
        "players": [
            {
                "id": row.player_id,
                "name": row.player_display_name,
                "team": row.team,
                "position": row.position
            }
            for _, row in results.iterrows()
        ]
    }

# -----------------------------------------------------------------------------
# /players/{player_id}/week/{week}
# -----------------------------------------------------------------------------
@router_players.get("/players/{player_id}/week/{week}")
def get_player_week(player_id: str, week: int):
    df = WEEKLY.copy()
    rows = df[(df["player_id"] == player_id)]

    if rows.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    # Weeks available
    wks = sorted([int(x) for x in rows["week"].dropna().unique()])

    # Find the weekly row
    selected = rows[rows["week"] == week]
    selected_row = selected.iloc[0] if not selected.empty else None

    # Basic player info
    player_info = {
        "id": player_id,
        "name": rows.iloc[0].player_display_name,
        "team": rows.iloc[0].team,
        "position": rows.iloc[0].position
    }

    # Schedule info
    match_info = get_match_info(player_info["team"], week)

    # Weekly log
    weekly_records = []
    for _, r in rows.sort_values("week").iterrows():
        weekly_records.append(r.to_dict())

    return {
        "ok": True,
        "player": player_info,
        "selectedWeek": week,
        "matchup": match_info,
        "weeksWithData": wks,
        "overview": selected_row.to_dict() if selected_row is not None else None,
        "weeklyLog": weekly_records
    }

# -----------------------------------------------------------------------------
# /player-props/week/{week} (Odds API)
# -----------------------------------------------------------------------------
@router_odds.get("/player-props/week/{week}")
def odds_for_week(week: int):
    if not ODDS_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Odds API key")

    url = (
        f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT}/odds/"
        f"?apiKey={ODDS_API_KEY}&regions={ODDS_API_REGION}&markets=player_pass_yds,player_rec_yds,player_rush_yds"
    )

    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    props_out = []

    for game in data:
        commence = game.get("commence_time")
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                market_key = market.get("key")

                for outcome in market.get("outcomes", []):
                    player = outcome.get("name")
                    line = outcome.get("point")

                    props_out.append({
                        "player": player,
                        "market": market_key,
                        "line": line,
                        "sportsbook": book.get("title"),
                        "game_time": commence
                    })

    return {"ok": True, "props": props_out}

# -----------------------------------------------------------------------------
# Simple AI projection: /ai/picks/{week}
# -----------------------------------------------------------------------------
@router_ai.get("/ai/picks/{week}")
def ai_pick_week(week: int):
    df = WEEKLY[WEEKLY["week"] == week]

    if df.empty:
        return {"ok": False, "error": "No data for this week"}

    picks = []

    for _, r in df.iterrows():
        score = (
            (r.get("passing_epa") or 0) +
            (r.get("rushing_epa") or 0) +
            (r.get("receiving_epa") or 0)
        )

        picks.append({
            "player": r.player_display_name,
            "team": r.team,
            "position": r.position,
            "score": round(score, 2)
        })

    picks = sorted(picks, key=lambda x: x["score"], reverse=True)

    return {"ok": True, "week": week, "picks": picks[:20]}

# -----------------------------------------------------------------------------
# REGISTER ROUTERS
# -----------------------------------------------------------------------------
app.include_router(router_players)
app.include_router(router_stats)
app.include_router(router_odds)
app.include_router(router_ai)

# -----------------------------------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "season": SEASON, "players_loaded": len(WEEKLY)}
