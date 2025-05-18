import pandas as pd


def extract_features(stats_df, home_team, away_team, round_num, season):
    # Find the row matching the input
    row = stats_df[
        (stats_df["Team"] == home_team)
        & (stats_df["Opponent"] == away_team)
        & (stats_df["Round"] == round_num)
        & (stats_df["Season"] == season)
    ]
    if row.empty:
        raise ValueError(
            "No match found for the given input. Check your team names, round, and season."
        )

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

    features = row[predictors]
    return features
