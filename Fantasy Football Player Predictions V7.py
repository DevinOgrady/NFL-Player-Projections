import pandas as pd
import numpy as np

df = pd.read_csv("weekly_player_stats_offense_smaller.csv")

df.columns = (df.columns.str.strip().str.lower())

df_2024 = df[(df["season"] == 2024) & (df['week'] < 18)].copy()

team_week = (df_2024.groupby(["week", "game_id", "team"],as_index=False).agg(
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

team_week["plays_per_game"] = team_week["total_plays"]

team_season = (team_week.groupby(["team"],as_index=False).agg
    (
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

player_week = (df_2024.groupby(["week","game_id","team","player_name","position"],as_index=False).agg(
        pass_attempts=("pass_attempts", "sum"),
        rush_attempts=("rush_attempts", "sum"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        offense_snaps=("offense_snaps", "sum")
    )
)

team_merge = team_week[["week","game_id","team","pass_attempts","rush_attempts","targets","offense_snaps"]].rename(
    columns={
        "pass_attempts": "team_pass_attempts",
        "rush_attempts": "team_rush_attempts",
        "targets": "team_targets",
        "offense_snaps": "team_offense_snaps"
    }
)


player_week = player_week.merge(team_merge,on=["week","game_id","team"],how="left")

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

player_season = (
    player_week
    .groupby(
        [
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

player_season = player_season.sort_values(
    [
        "team",
        "position",
        "avg_snap_share",
        "avg_target_share",
        "touches_per_game"
    ],
    ascending=[
        True,
        True,
        False,
        False,
        False
    ]
).reset_index(drop=True)

role_usage = (
    player_season
    .groupby(
        [
            "team",
            "position",
            "depth_rank_2024",
            "depth_role_2024"
        ],
        as_index=False
    )
    .agg(

        total_targets=("targets", "sum"),
        total_rush_attempts=("rush_attempts", "sum"),
        total_touches=("touches", "sum"),

        avg_target_share=("avg_target_share", "mean"),
        avg_rush_attempt_share=("avg_rush_attempt_share", "mean"),
        avg_snap_share=("avg_snap_share", "mean"),

        avg_targets_per_game=("targets_per_game", "mean"),
        avg_rush_attempts_per_game=("rush_attempts_per_game", "mean"),
        avg_touches_per_game=("touches_per_game", "mean"),
    )
)

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

role_usage["role_target_share_2024"] = np.where(
    role_usage["team_targets"] > 0,
    role_usage["total_targets"] /
    role_usage["team_targets"],
    0
)

role_usage["role_rush_share_2024"] = np.where(
    role_usage["team_rush_attempts"] > 0,
    role_usage["total_rush_attempts"] /
    role_usage["team_rush_attempts"],
    0
)

depth_2024 = player_season[
    [
        "team",
        "player_name",
        "position",
        "depth_rank_2024",
        "depth_role_2024",

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
    ]
].copy()

with pd.ExcelWriter(
    "output.xlsx",
    engine="openpyxl"
) as writer:

    team_week.to_excel(
        writer,
        sheet_name="2024_Team_Weekly",
        index=False
    )

    team_season.to_excel(
        writer,
        sheet_name="2024_Team_Season",
        index=False
    )

    player_week.to_excel(
        writer,
        sheet_name="2024_Player_Weekly",
        index=False
    )

    player_season.to_excel(
        writer,
        sheet_name="2024_Player_Season",
        index=False
    )

    depth_2024.to_excel(
        writer,
        sheet_name="2024_Depth_Usage",
        index=False
    )

    role_usage.to_excel(
        writer,
        sheet_name="2024_Role_Usage",
        index=False
    )
