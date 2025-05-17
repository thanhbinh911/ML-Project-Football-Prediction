import streamlit as st
import requests

st.title("🏟️ Football Match Result Predictor")

team = st.text_input("Home Team")
opponent = st.text_input("Away Team")
round_num = st.number_input("Round", min_value=1, max_value=38, step=1)
season = st.number_input("Season", min_value=2016, max_value=2025, step=1, value=2023)

if st.button("Predict"):
    data = {"Team": team, "Opponent": opponent, "Round": round_num, "Season": season}
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        result = response.json().get("prediction", "error")
        st.success(f"Predicted Result: {result}")
    except Exception as e:
        st.error(f"Error connecting to prediction service: {e}")
