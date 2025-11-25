from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import nflreadpy as nfl
import pandas as pd
import polars as pl
from functools import lru_cache

SEASON = 2025

app = FastAPI(
    title="NFL Props API",
    description="Simple API that exposes nflverse weekly 2025 player stats",
    version="0.1.2",
)

# Allow your frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_weekly_stats() -> pd.DataFrame:
    """
    Load nflverse weekly player-level stats for the current season.

    nflreadpy returns a Polars DataFrame by default.
    We will:
      1) Filter positions using Polars syntax (no .isin)
      2) Convert the result to pandas
    """
    # Polars DataFrame
    pl_df = nfl.load_player_stats(
        seasons=[SEASON],
        summary_level="week",
    )

    # Filter to offensive skill positions (QB/RB/WR/TE) using Polars
    pl_df = pl_df.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))

    # Convert to pandas DataFrame for the rest of the code
    df = pl_df.to_pandas()

    return df


@app.get("/health")
def health():
    """
    Simple health check to make sure stats loaded correctly.
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


@app.get("/players")
def list_players(q: str | None = None):
    """
    Return a unique list of players (id, name, team, position).
    Optional q = case-insensitive substring match on name.
    """
    df = load_weekly_stats()

    # Only keep the columns we care about and deduplicate players
    cols = ["player_id", "player_display_name", "recent_team", "position"]
    base = df[cols].drop_duplicates()

    if q:
        q_lower = q.lower()
        mask = base["player_display_name"].str.lower().str.contains(q_lower)
        base = base[mask]

    players = [
        {
            "id": row.player_id,
            "name": row.player_display_name,
            "team": row.recent_team,
            "position": row.position,
        }
        for row in base.itertuples()
    ]

    return {"players": players}


@app.get("/player/{player_id}")
def player_detail(player_id: str, week: int | None = None):
    """
    Return stats for a single player.

    - week: specific week (if None, return latest week with data)
    Also includes a small weekly log (recent games).
    """
    df = load_weekly_stats()

    pdf = df[df["player_id"] == player_id].copy()
    if pdf.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    # Basic identity info
    first = pdf.iloc[0]
    identity = {
        "id": first["player_id"],
        "name": first["player_display_name"],
        "team": first["recent_team"],
        "position": first["position"],
    }

    # Determine which week to show
    if week is None:
        latest_week = int(pdf["week"].max())
    else:
        latest_week = int(week)

    # Single week row
    row = pdf[pdf["week"] == latest_week]
    if row.empty:
        overview = None
    else:
        r = row.iloc[0]
        overview = {
            "week": int(r["week"]),
            "season_type": r["season_type"],
            "team": r["recent_team"],
            "opponent": r.get("opponent_team", None),
            "passing_yards": r.get("passing_yards", None),
            "passing_tds": r.get("passing_tds", None),
            "interceptions": r.get("interceptions", None),
            "rushing_yards": r.get("rushing_yards", None),
            "rushing_tds": r.get("rushing_tds", None),
            "receptions": r.get("receptions", None),
            "receiving_yards": r.get("receiving_yards", None),
            "receiving_tds": r.get("receiving_tds", None),
        }

    # Recent weekly log (last 5 weeks with data)
    pdf_sorted = pdf.sort_values("week", ascending=False).head(5)
    weekly_log = []
    for r in pdf_sorted.itertuples():
        weekly_log.append(
            {
                "week": int(r.week),
                "team": r.recent_team,
                "opponent": getattr(r, "opponent_team", None),
                "passing_yards": getattr(r, "passing_yards", None),
                "passing_tds": getattr(r, "passing_tds", None),
                "interceptions": getattr(r, "interceptions", None),
                "rushing_yards": getattr(r, "rushing_yards", None),
                "rushing_tds": getattr(r, "rushing_tds", None),
                "receptions": getattr(r, "receptions", None),
                "receiving_yards": getattr(r, "receiving_yards", None),
                "receiving_tds": getattr(r, "receiving_tds", None),
            }
        )

    return {
        "player": identity,
        "selectedWeek": latest_week,
        "overview": overview,
        "weeklyLog": weekly_log,
    }


# Local dev entrypoint (not used on Render, but nice to have)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
