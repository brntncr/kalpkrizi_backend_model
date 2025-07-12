from fastapi import FastAPI, Depends
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import os
from sqlalchemy import Column, Integer, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

app = FastAPI()

# 📦 Model ve yardımcı dosyaları yükleme
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "feature_order.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    feature_order = pickle.load(f)

NUMERIC_COLS = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']

# 🔹 Railway Postgres DB bağlantısı (env’den alıyoruz)
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

# 🔹 Railway deploy sonrası ilk çalışmada tabloyu yaratır
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

@app.get("/")
async def root():
    return {"message": "Heart attack prediction API is running on Railway!"}

@app.post("/predict")
async def predict(data: HeartAttackInput, db: Session = Depends(get_db)):
    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])
    df_encoded = pd.get_dummies(df_input, columns=CATEGORICAL_COLS, drop_first=True)

    for col in feature_order:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[feature_order]

    numeric_features = [col for col in df_encoded.columns if any(n in col for n in NUMERIC_COLS)]
    df_encoded[numeric_features] = scaler.transform(df_encoded[numeric_features])

    pred = model.predict(df_encoded)[0]

    record = PredictionRecord(
        age=data.age, sex=data.sex, cp=data.cp, trtbps=data.trtbps, chol=data.chol,
        fbs=data.fbs, restecg=data.restecg, thalachh=data.thalachh, exng=data.exng,
        oldpeak=data.oldpeak, slp=data.slp, caa=data.caa, thall=data.thall,
        prediction=int(pred)
    )
    db.add(record)
    db.commit()

    return {"prediction": int(pred)}
