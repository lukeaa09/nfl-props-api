from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nflreadpy as nfl
import pandas as pd
import polars as pl
from functools import lru_cache

SEASON = 2025

app = FastAPI(
    title="NFL Props API",
    description="Simple API that exposes nflverse weekly player stats",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_weekly_stats() -> pd.DataFrame:
    """
    Load nflverse weekly player stats for one season.

    nflreadpy returns a Polars DataFrame by default.
    We always convert to pandas BEFORE doing any filtering.
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
        # Fallback, just in case
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
        return {
            "ok": True,
            "columns": list(df.columns),
            "sample": df.head(3).to_dict(orient="records"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/players")
def list_players(q: str | None = None):
    """
    List players (id, name, team, position) for offensive positions.
    Optional q: case-insensitive substring match on name.
    """
    try:
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

        # Filter to offensive skill positions only
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

    except Exception as e:
        return {
            "ok": False,
            "error": f"list_players_failed: {e}",
        }


@app.get("/player/{player_id}")
def player_detail(player_id: str, week: int | None = None):
    """
    Detailed stats for one player.

    - If `week` is omitted, use the latest week with data.
    - Returns:
        ok: bool
        player: { id, name, team, position }
        selectedWeek: int
        weeksWithData: [int, ...]
        overview: dict for the selected week (or null if no row)
        weeklyLog: list of dicts for all weeks for that player
    """
    try:
        df = load_weekly_stats()

        # Basic column sanity
        cols = set(df.columns)
        if "player_id" not in cols or "week" not in cols:
            return {
                "ok": False,
                "error": "Missing 'player_id' or 'week' column",
                "columns": sorted(cols),
            }

        # All rows for this player
        pdf = df[df["player_id"] == player_id].copy()
        if pdf.empty:
            return {"ok": False, "error": "Player not found"}

        # Weeks with data
        weeks_with_data = sorted(int(w) for w in pdf["week"].unique())

        # Decide which week to use
        if week is None:
            selected_week = max(weeks_with_data)
        else:
            selected_week = int(week)

        row_df = pdf[pdf["week"] == selected_week]
        if row_df.empty:
            overview = None
        else:
            # Convert whole row to a plain dict of stats
            overview = row_df.iloc[0].to_dict()

        # Full weekly log (all weeks for this player)
        pdf_sorted = pdf.sort_values("week")
        weekly_log = pdf_sorted.to_dict(orient="records")

        # Player identity (use any row)
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

        return {
            "ok": True,
            "player": identity,
            "selectedWeek": selected_week,
            "weeksWithData": weeks_with_data,
            "overview": overview,
            "weeklyLog": weekly_log,
        }

    except Exception as e:
        # Catch EVERYTHING so the client never sees a 500 HTML page
        return {
            "ok": False,
            "error": f"player_detail_failed: {e}",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
