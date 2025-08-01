from fastapi import FastAPI, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import os
import logging
import requests
from sqlalchemy import Column, Integer, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from app.custom_transformers import FeatureEngineer

# Initialize FastAPI app
app = FastAPI()

# Setup CORS (keep permissive for now; restrict for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 📦 Model files and helpers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "..", "models", "pipeline.pkl")

with open(PIPELINE_PATH, "rb") as f:
    pipeline = pickle.load(f)

# 🔹 DB setup
DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL is not None, "DATABASE_URL environment variable must be set!"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    age = Column(Integer)
    sex = Column(Integer)
    cp = Column(Integer)
    trtbps = Column(Integer)
    chol = Column(Integer)
    fbs = Column(Integer)
    restecg = Column(Integer)
    thalachh = Column(Integer)
    exng = Column(Integer)
    oldpeak = Column(Float)
    slp = Column(Integer)
    caa = Column(Integer)
    thall = Column(Integer)
    prediction = Column(Integer)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class HeartAttackInput(BaseModel):
    age: int
    sex: int
    cp: int
    trtbps: int
    chol: int
    fbs: int
    restecg: int
    thalachh: int
    exng: int
    oldpeak: float
    slp: int
    caa: int
    thall: int

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY is not None, "GOOGLE_API_KEY must be set!"

# 🔹 Routes

@app.get("/")
async def root():
    return {"message": "Unified backend with /predict + /ask-ai running on Railway!"}

@app.post("/predict")
async def predict(
    data: HeartAttackInput,
    explain: bool = Query(False, description="Set to true for Gemini explanation"),
    db: Session = Depends(get_db)
):
    logger.info(f"Received predict request: {data.dict()}")

    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])

    # 🔄 Artık preprocessing ve model tek bir pipeline içinde
    pred = pipeline.predict(df_input)[0]
    pred_proba = pipeline.predict_proba(df_input)[0][1]

    # DB kaydı yapmak istersen bu alanı aktifleştir
    # record = PredictionRecord(**input_dict, prediction=int(pred))
    # db.add(record)
    # db.commit()

    logger.info(f"Prediction: {pred} (probability: {pred_proba:.2%})")

    outcome = "Yüksek Risk" if pred == 1 else "Düşük Risk"
    message = (
        f"Yüksek kalp krizi riski tespit edildi (Güven: {pred_proba:.2%})"
        if pred == 1 else
        f"Düşük kalp krizi riski tahmin edildi (Güven: {1 - pred_proba:.2%})"
    )

    explanation = None
    if explain:
        prompt = (
            f"Kullanıcının sağlık verileri şu şekildedir: {input_dict}.\n"
            f"Makine öğrenmesi modeli kalp krizi riskini: {outcome} olarak tahmin etti.\n"
            f"Lütfen bu sonucu basit bir dille açıklayın ve 3 genel yaşam önerisi verin.\n"
            f"Cevabını Türkçe olarak yaz.\n"
            f"Lütfen herhangi bir emoji kullanma ve direk açıklamaya geç."
        )
        explanation = call_gemini_api(prompt)

    return {
        "prediction": int(pred),
        "prediction_probability": round(float(pred_proba), 4),
        "outcome_message": message,
        "explanation": explanation
    }

@app.post("/ask-ai")
async def ask_ai(payload: dict = Body(...)):
    user_prompt = payload.get("question")
    if not user_prompt:
        return {"error": "question field is required."}

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": GOOGLE_API_KEY}
    body = {"contents": [{"parts": [{"text": user_prompt}]}]}

    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        res_json = response.json()
        text = res_json['candidates'][0]['content']['parts'][0]['text']
        return {"answer": text}
    else:
        logger.error(f"Gemini API error {response.status_code}: {response.text}")
        return {"error": f"Gemini API error {response.status_code}"}

# 🔹 Helper function for Gemini API (used by /predict too)
def call_gemini_api(prompt: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": GOOGLE_API_KEY}
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        res_json = response.json()
        return res_json['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Gemini API error: {response.status_code}"
