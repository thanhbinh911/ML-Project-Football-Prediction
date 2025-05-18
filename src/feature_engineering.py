import pandas as pd
import numpy as np


# Load data
matches = pd.read_excel("data/football_data.xlsx")


# Team name mapping
class MisssingDict(dict):
    __missing__ = lambda self, key: key


map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Huddersfield Town": "Huddersfield",
    "West Bromwich Albion": "West Brom",
    "Nottingham Forest": "Nott'ham Forest",
    "Sheffield United": "Sheffield Utd",
    "Newcastle United": "Newcastle Utd",
}

mapping = MisssingDict(map_values)
matches["Team"] = matches["Team"].map(mapping)
matches["Round"] = matches["Round"].str.extract(r"(\d+)").astype(int)


matches["Date"] = pd.to_datetime(matches["Date"])
matches["Venue_Code"] = matches["Venue"].astype("category").cat.codes
matches = matches.sort_values(["Team", "Season", "Date"], ascending=[True, True, True])
matches.index = range(matches.shape[0])

columns_to_average = ["Sh", "SoT", "Dist", "GF", "GA", "xG", "xGA", "Poss"]
rolling_columns = [f"{col}_rolling_5" for col in columns_to_average]
for col, rolling_col in zip(columns_to_average, rolling_columns):
    matches[rolling_col] = (
        matches.groupby(["Team", "Season"])[col]
        .rolling(5, min_periods=1, closed="left")
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

team_rolling_stats = matches[["Opponent", "Season", "Date"] + rolling_columns].copy()
team_rolling_stats = team_rolling_stats.rename(
    columns={col: f"Opponent_{col}" for col in rolling_columns}
)
matches = matches.merge(
    team_rolling_stats,
    left_on=["Team", "Season", "Date"],
    right_on=["Opponent", "Season", "Date"],
    suffixes=("", "_opponent"),
)
matches.drop(columns="Opponent_opponent", inplace=True)

# Compute difference columns
difference_columns = [f"{col}_diff" for col in columns_to_average]
for col, diff_col in zip(rolling_columns, difference_columns):
    matches[diff_col] = matches[col] - matches[f"Opponent_{col}"]

cols_to_round = ["Sh_diff", "xG_diff", "xGA_diff"]
matches[cols_to_round] = matches[cols_to_round].round(2)


matches["Opp_Code"] = matches["Opponent"].astype("category").cat.codes
matches["Hour"] = matches["Time"].str.extract(r"(\d{2}:\d{2})")
matches["Hour"] = matches["Hour"].str.replace(r":.+", "", regex=True).astype(int)
matches["day_code"] = matches["Date"].dt.dayofweek
matches["target"] = (matches["Result"] == "W").astype(int)

matches["is_unbeaten"] = matches["Result"].isin(["W", "D"])
matches["is_unbeaten_shifted"] = matches.groupby(["Team", "Season"])[
    "is_unbeaten"
].shift(1)


def compute_unbeaten_streak(group):
    streak = 0
    streaks = []
    for unbeaten in group["is_unbeaten_shifted"]:
        if unbeaten is np.nan:
            streak = 0
        elif unbeaten:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    return streaks


matches["unbeaten_streak"] = (
    matches.groupby(["Team", "Season"])
    .apply(lambda x: pd.Series(compute_unbeaten_streak(x), index=x.index))
    .reset_index(level=[0, 1], drop=True)
)
matches = matches.drop(columns=["is_unbeaten_shifted"])
matches.rename(columns={"unbeaten_streak": "unbeaten_streak_home"}, inplace=True)

unbeaten_streak_temp = matches[
    ["Opponent", "Season", "Date", "unbeaten_streak_home"]
].copy()
unbeaten_streak_temp.rename(
    columns={"unbeaten_streak_home": "unbeaten_streak_away"}, inplace=True
)
matches = matches.merge(
    unbeaten_streak_temp,
    left_on=["Team", "Season", "Date"],
    right_on=["Opponent", "Season", "Date"],
)
matches.drop(columns="Opponent_y", inplace=True)
matches.rename(columns={"Opponent_x": "Opponent"}, inplace=True)
matches["unbeaten_streak_diff"] = (
    matches["unbeaten_streak_home"] - matches["unbeaten_streak_away"]
)

# Previous season ranking
team_ranks = matches.drop_duplicates(subset=["Season", "Team"]).copy()
team_ranks["Season_Rank"] = team_ranks.groupby("Season").cumcount() + 1
previous_season_ranks = team_ranks[["Season", "Team", "Season_Rank"]].copy()
previous_season_ranks["Season"] += 1
matches = matches.merge(previous_season_ranks, on=["Season", "Team"], how="left")
matches.rename(columns={"Season_Rank": "home_ranking_last_season"}, inplace=True)
matches["home_ranking_last_season"] = matches["home_ranking_last_season"].fillna(-1)

# manually handle the first season in the dataset
earliest_season = matches["Season"].min()
previous_season_rankings = {
    "Arsenal": 3,
    "Leicester City": 14,
    "Tottenham": 5,
    "Manchester City": 2,
    "Manchester Utd": 4,
    "Southampton": 7,
    "West Ham": 12,
    "Liverpool": 6,
    "Stoke City": 9,
    "Chelsea": 1,
    "Everton": 11,
    "Swansea City": 8,
    "Watford": -1,
    "West Brom": 13,
    "Crystal Palace": 10,
    "Bournemouth": -1,
    "Sunderland": 16,
    "Newcastle Utd": 15,
    "Norwich City": -1,
    "Aston Villa": 17,
}

matches.loc[matches["Season"] == earliest_season, "home_ranking_last_season"] = (
    matches.loc[matches["Season"] == earliest_season, "Team"].map(
        previous_season_rankings
    )
)
matches["home_ranking_last_season"] = matches["home_ranking_last_season"].astype(int)

previous_season_ranks = matches[
    ["Opponent", "Season", "Date", "home_ranking_last_season"]
].copy()
previous_season_ranks.rename(
    columns={"home_ranking_last_season": "away_ranking_last_season"}, inplace=True
)
matches = matches.merge(
    previous_season_ranks,
    left_on=["Team", "Season", "Date"],
    right_on=["Opponent", "Season", "Date"],
)
matches.rename(columns={"Opponent_x": "Opponent"}, inplace=True)
matches.drop(columns=["Opponent_y"], inplace=True)
matches["LastSeasonRank"] = (
    matches["home_ranking_last_season"] - matches["away_ranking_last_season"]
)

# PromotedMatchup
matches["PromotedMatchup"] = 0
matches.loc[
    (matches["home_ranking_last_season"] == -1)
    & (matches["away_ranking_last_season"] != -1),
    "PromotedMatchup",
] = 1
matches.loc[
    (matches["away_ranking_last_season"] == -1)
    & (matches["home_ranking_last_season"] != -1),
    "PromotedMatchup",
] = -1

# Prepare training data
predictors = [
    "SoT_diff",
    "Poss_diff",
    "unbeaten_streak_diff",
    "GA_diff",
    "GF_diff",
    "xG_diff",
    "xGA_diff",
    "LastSeasonRank",
    "PromotedMatchup",
]

matchesB = matches[
    [
        "Round",
        "Season",
        "Date",
        "Team",
        "Opponent",
        "Venue",
        "Result",
        "Sh_diff",
        "SoT_diff",
        "GF_diff",
        "GA_diff",
        "xG_diff",
        "xGA_diff",
        "Poss_diff",
        "unbeaten_streak_diff",
        "LastSeasonRank",
        "PromotedMatchup",
        "target",
    ]
]
matchesB = matchesB[matchesB["Venue"] == "Home"]
matchesB.dropna(
    subset=[
        "Sh_diff",
        "SoT_diff",
        "GF_diff",
        "GA_diff",
        "xG_diff",
        "xGA_diff",
        "Poss_diff",
    ],
    inplace=True,
)
