import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "weekly_player_stats_offense.csv"
OUTPUT_FILE = "NFL_2025_Usage_Baseline.xlsx"

SEASON = 2025

SKILL_POSITIONS = ["QB", "RB", "FB", "WR", "TE"]
TARGET_POSITIONS = ["RB", "FB", "WR", "TE"]
RUSH_POSITIONS = ["QB", "RB", "FB"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(series):
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    )


def clean_team(series):
    return (
        series.astype("string")
        .str.upper()
        .str.strip()
    )


def clean_position(series):
    return (
        series.astype("string")
        .str.upper()
        .str.strip()
    )


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)


# ============================================================
# CLEAN IMPORTANT COLUMNS
# ============================================================

if "player_name" not in df.columns:
    raise KeyError("Column 'player_name' was not found.")

if "team" not in df.columns:
    raise KeyError("Column 'team' was not found.")

if "position" not in df.columns:
    raise KeyError("Column 'position' was not found.")

if "season" not in df.columns:
    raise KeyError("Column 'season' was not found.")

if "week" not in df.columns:
    raise KeyError("Column 'week' was not found.")

if "game_id" not in df.columns:
    raise KeyError("Column 'game_id' was not found.")


df["player_name"] = clean_text(df["player_name"])
df["team"] = clean_team(df["team"])
df["position"] = clean_position(df["position"])

df["season"] = pd.to_numeric(df["season"], errors="coerce")
df["week"] = pd.to_numeric(df["week"], errors="coerce")


# ============================================================
# KEEP ONLY 2025
# ============================================================

df_2024 = df[(df["season"] == 2025) & (df['week'] < 18)].copy()

df = df.drop_duplicates().reset_index(drop=True)

positions_to_drop = [
    "P",
    "SAF",
    "LS",
    "G",
    "OT",
    "CB",
    "DE",
    "DT",
    "C"
]

df = df[~df["position"].isin(positions_to_drop)]

# ============================================================
# REQUIRED STAT COLUMNS
# ============================================================

STAT_COLUMNS = [
    "pass_attempts",
    "rush_attempts",
    "targets",
    "receptions",
    "receiving_yards",
    "rushing_yards",
    "offense_snaps"
]

for col in STAT_COLUMNS:
    if col not in df.columns:
        df[col] = 0

df = safe_numeric(df, STAT_COLUMNS)


# ============================================================
# 1. TEAM WEEKLY USAGE
# ============================================================

team_week = (
    df
    .groupby(
        ["season", "week", "game_id", "team"],
        as_index=False
    )
    .agg(
        pass_attempts=("pass_attempts", "sum"),
        rush_attempts=("rush_attempts", "sum"),
        targets=("targets", "sum"),
        offense_snaps=("offense_snaps", "max")
    )
)

team_week["total_plays"] = (
    team_week["pass_attempts"] +
    team_week["rush_attempts"]
)

team_week["pass_pct"] = np.where(
    team_week["total_plays"] > 0,
    team_week["pass_attempts"] /
    team_week["total_plays"],
    0
)

team_week["rush_pct"] = np.where(
    team_week["total_plays"] > 0,
    team_week["rush_attempts"] /
    team_week["total_plays"],
    0
)

team_week["targets_per_pass_attempt"] = np.where(
    team_week["pass_attempts"] > 0,
    team_week["targets"] /
    team_week["pass_attempts"],
    0
)

team_week["plays_per_game"] = team_week["total_plays"]


# ============================================================
# 2. TEAM SEASON USAGE
# ============================================================

team_season = (
    team_week
    .groupby(
        ["season", "team"],
        as_index=False
    )
    .agg(
        games=("game_id", "nunique"),
        pass_attempts=("pass_attempts", "sum"),
        rush_attempts=("rush_attempts", "sum"),
        targets=("targets", "sum"),
        total_plays=("total_plays", "sum")
    )
)

team_season["pass_pct"] = np.where(
    team_season["total_plays"] > 0,
    team_season["pass_attempts"] /
    team_season["total_plays"],
    0
)

team_season["rush_pct"] = np.where(
    team_season["total_plays"] > 0,
    team_season["rush_attempts"] /
    team_season["total_plays"],
    0
)

team_season["plays_per_game"] = np.where(
    team_season["games"] > 0,
    team_season["total_plays"] /
    team_season["games"],
    0
)

team_season["targets_per_game"] = np.where(
    team_season["games"] > 0,
    team_season["targets"] /
    team_season["games"],
    0
)

team_season["pass_attempts_per_game"] = np.where(
    team_season["games"] > 0,
    team_season["pass_attempts"] /
    team_season["games"],
    0
)

team_season["rush_attempts_per_game"] = np.where(
    team_season["games"] > 0,
    team_season["rush_attempts"] /
    team_season["games"],
    0
)

team_season["targets_per_pass_attempt"] = np.where(
    team_season["pass_attempts"] > 0,
    team_season["targets"] /
    team_season["pass_attempts"],
    0
)


# ============================================================
# 3. PLAYER WEEKLY USAGE
# ============================================================

player_week = (
    df
    .groupby(
        [
            "season",
            "week",
            "game_id",
            "team",
            "player_name",
            "position"
        ],
        as_index=False
    )
    .agg(
        pass_attempts=("pass_attempts", "sum"),
        rush_attempts=("rush_attempts", "sum"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        offense_snaps=("offense_snaps", "sum")
    )
)


# ============================================================
# MERGE TEAM WEEKLY TOTALS
# ============================================================

team_merge = team_week[
    [
        "season",
        "week",
        "game_id",
        "team",
        "pass_attempts",
        "rush_attempts",
        "targets",
        "offense_snaps"
    ]
].rename(
    columns={
        "pass_attempts": "team_pass_attempts",
        "rush_attempts": "team_rush_attempts",
        "targets": "team_targets",
        "offense_snaps": "team_offense_snaps"
    }
)


player_week = player_week.merge(
    team_merge,
    on=[
        "season",
        "week",
        "game_id",
        "team"
    ],
    how="left"
)


# ============================================================
# PLAYER SHARES
# ============================================================

player_week["target_share"] = np.where(
    player_week["team_targets"] > 0,
    player_week["targets"] /
    player_week["team_targets"],
    0
)

player_week["rush_attempt_share"] = np.where(
    player_week["team_rush_attempts"] > 0,
    player_week["rush_attempts"] /
    player_week["team_rush_attempts"],
    0
)

player_week["snap_share"] = np.where(
    player_week["team_offense_snaps"] > 0,
    player_week["offense_snaps"] /
    player_week["team_offense_snaps"],
    0
)

player_week["touches"] = (
    player_week["rush_attempts"] +
    player_week["receptions"]
)


# ============================================================
# 4. PLAYER SEASON USAGE
# ============================================================

player_season = (
    player_week
    .groupby(
        [
            "season",
            "team",
            "player_name",
            "position"
        ],
        as_index=False
    )
    .agg(
        games=("game_id", "nunique"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        rush_attempts=("rush_attempts", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        offense_snaps=("offense_snaps", "sum"),
        avg_target_share=("target_share", "mean"),
        avg_rush_attempt_share=("rush_attempt_share", "mean"),
        avg_snap_share=("snap_share", "mean"),
        avg_targets=("targets", "mean"),
        avg_rush_attempts=("rush_attempts", "mean"),
        avg_touches=("touches", "mean")
    )
)


# ============================================================
# PLAYER USAGE METRICS
# ============================================================

player_season["touches"] = (
    player_season["rush_attempts"] +
    player_season["receptions"]
)

player_season["opportunities"] = (
    player_season["targets"] +
    player_season["rush_attempts"]
)

player_season["targets_per_game"] = np.where(
    player_season["games"] > 0,
    player_season["targets"] /
    player_season["games"],
    0
)

player_season["rush_attempts_per_game"] = np.where(
    player_season["games"] > 0,
    player_season["rush_attempts"] /
    player_season["games"],
    0
)

player_season["touches_per_game"] = np.where(
    player_season["games"] > 0,
    player_season["touches"] /
    player_season["games"],
    0
)

player_season["opportunities_per_game"] = np.where(
    player_season["games"] > 0,
    player_season["opportunities"] /
    player_season["games"],
    0
)

player_season["catch_rate"] = np.where(
    player_season["targets"] > 0,
    player_season["receptions"] /
    player_season["targets"],
    0
)

player_season["yards_per_target"] = np.where(
    player_season["targets"] > 0,
    player_season["receiving_yards"] /
    player_season["targets"],
    0
)

player_season["yards_per_reception"] = np.where(
    player_season["receptions"] > 0,
    player_season["receiving_yards"] /
    player_season["receptions"],
    0
)

player_season["yards_per_carry"] = np.where(
    player_season["rush_attempts"] > 0,
    player_season["rushing_yards"] /
    player_season["rush_attempts"],
    0
)


# ============================================================
# 5. CREATE 2025 DEPTH ROLES
# ============================================================
#
# The role score determines the player's role within:
#
# Team + Position
#
# Higher usage = higher depth role.
#
# Example:
#
# HOU WR:
# Nico Collins       -> WR1
# Player B           -> WR2
# Player C           -> WR3
#
# ============================================================

player_season["snap_score"] = (
    player_season["avg_snap_share"] * 100
)

player_season["target_score"] = (
    player_season["avg_target_share"] * 100
)

player_season["rush_score"] = (
    player_season["avg_rush_attempt_share"] * 100
)

player_season["touch_score"] = (
    player_season["touches_per_game"]
)


# Role score
player_season["role_score"] = (
    0.40 * player_season["snap_score"] +
    0.30 * player_season["target_score"] +
    0.20 * player_season["rush_score"] +
    0.10 * player_season["touch_score"]
)


# ============================================================
# DEPTH RANK
# ============================================================

player_season = player_season.sort_values(
    [
        "team",
        "position",
        "role_score",
        "avg_snap_share",
        "avg_target_share",
        "touches_per_game"
    ],
    ascending=[
        True,
        True,
        False,
        False,
        False,
        False
    ]
).reset_index(drop=True)


player_season["depth_rank_2025"] = (
    player_season
    .groupby(["team", "position"])
    .cumcount() + 1
)


player_season["depth_role_2025"] = (
    player_season["position"] +
    player_season["depth_rank_2025"].astype(str)
)


# ============================================================
# 6. ROLE-LEVEL USAGE
# ============================================================
#
# This is the important table for the future 2026 model.
#
# Instead of saying:
#
# "Nico Collins owns 24% of Houston's targets"
#
# we can say:
#
# "Houston's WR1 role historically produced X% of
#  Houston's targets."
#
# This allows the 2026 WR1 to inherit the role regardless
# of who occupies it.
#
# ============================================================

role_usage = (
    player_season
    .groupby(
        [
            "team",
            "position",
            "depth_rank_2025",
            "depth_role_2025"
        ],
        as_index=False
    )
    .agg(
        players=("player_name", "count"),

        total_targets=("targets", "sum"),
        total_rush_attempts=("rush_attempts", "sum"),
        total_touches=("touches", "sum"),

        avg_target_share=("avg_target_share", "mean"),
        avg_rush_attempt_share=("avg_rush_attempt_share", "mean"),
        avg_snap_share=("avg_snap_share", "mean"),

        avg_targets_per_game=("targets_per_game", "mean"),
        avg_rush_attempts_per_game=("rush_attempts_per_game", "mean"),
        avg_touches_per_game=("touches_per_game", "mean"),

        avg_role_score=("role_score", "mean")
    )
)


# ============================================================
# TEAM TOTALS FOR ROLE SHARE CALCULATIONS
# ============================================================

team_totals = (
    player_season
    .groupby("team", as_index=False)
    .agg(
        team_targets=("targets", "sum"),
        team_rush_attempts=("rush_attempts", "sum")
    )
)


role_usage = role_usage.merge(
    team_totals,
    on="team",
    how="left"
)


# ============================================================
# ROLE TARGET / RUSH SHARE
# ============================================================

role_usage["role_target_share_2025"] = np.where(
    role_usage["team_targets"] > 0,
    role_usage["total_targets"] /
    role_usage["team_targets"],
    0
)

role_usage["role_rush_share_2025"] = np.where(
    role_usage["team_rush_attempts"] > 0,
    role_usage["total_rush_attempts"] /
    role_usage["team_rush_attempts"],
    0
)


# ============================================================
# 7. FINAL PLAYER DEPTH TABLE
# ============================================================

depth_2025 = player_season[
    [
        "team",
        "player_name",
        "position",
        "depth_rank_2025",
        "depth_role_2025",

        "games",

        "targets",
        "receptions",
        "rush_attempts",
        "touches",
        "opportunities",

        "targets_per_game",
        "rush_attempts_per_game",
        "touches_per_game",
        "opportunities_per_game",

        "avg_target_share",
        "avg_rush_attempt_share",
        "avg_snap_share",

        "catch_rate",
        "yards_per_target",
        "yards_per_reception",
        "yards_per_carry",

        "role_score"
    ]
].copy()

# ============================================================
# 9. SAVE EXCEL WORKBOOK
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    team_week.to_excel(
        writer,
        sheet_name="2025_Team_Weekly",
        index=False
    )

    team_season.to_excel(
        writer,
        sheet_name="2025_Team_Season",
        index=False
    )

    player_week.to_excel(
        writer,
        sheet_name="2025_Player_Weekly",
        index=False
    )

    player_season.to_excel(
        writer,
        sheet_name="2025_Player_Season",
        index=False
    )

    depth_2025.to_excel(
        writer,
        sheet_name="2025_Depth_Usage",
        index=False
    )

    role_usage.to_excel(
        writer,
        sheet_name="2025_Role_Usage",
        index=False
    )