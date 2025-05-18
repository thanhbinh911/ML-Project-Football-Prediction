from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
from src.extractor import extract_features
from src.feature_engineering import matchesB


app = FastAPI()
templates = Jinja2Templates(directory="Template")
app.mount("/static", StaticFiles(directory="Static"), name="static")
model = joblib.load("models/football_rf_model.pkl")


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_result(
    request: Request,
    Team: str = Form(...),
    Opponent: str = Form(...),
    Round: int = Form(...),
    Season: int = Form(...),
):
    try:
        features = extract_features(matchesB, Team, Opponent, Round, Season)
        prediction = model.predict(features)[0]
        result = "Win" if prediction == 1 else "Non-Win"
        return templates.TemplateResponse(
            "result.html", {"request": request, "prediction": result}
        )
    except Exception as e:
        error_msg = f"Error: {str(e)}. Please check your inputs."
        return templates.TemplateResponse(
            "result.html", {"request": request, "prediction": error_msg}
        )
