from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from train_model import matchesB
from extractor import extract_features


app = FastAPI()
model = joblib.load("football_rf_model.pkl")


class MatchData(BaseModel):
    Team: object
    Opponent: object
    Round: int
    Season: int


@app.post("/predict")
def predict_match(data: MatchData):
    features = extract_features(
        matchesB, data.Team, data.Opponent, data.Round, data.Season
    )
    prediction = model.predict([features])[0]
    result = "win" if prediction == 1 else "non_win"
    return {"prediction": result}
