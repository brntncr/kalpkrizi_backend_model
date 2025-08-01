import pandas as pd
import pickle
import sys
import os

# ✅ Yol ayarı (app import'u için)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from app.custom_transformers import FeatureEngineer

# 🔹 Kullanılacak sütunlar
numeric_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak", 
                "exng_oldpeak_carpim", "thalach_age_orani", "risk_skoru_light"]

categorical_cols = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall", 
                    "yas_grubu", "oldpeak_yuksek", "egim_flat_veya_down", 
                    "sessiz_gogus_agrisi", "thal_geri_donen_defekt"]

# 🔹 Preprocessing adımı
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
])

# 🔹 Pipeline tanımı
full_pipeline = Pipeline(steps=[
    ("features", FeatureEngineer()),
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(
        C=0.1,
        l1_ratio=0.25,
        penalty='elasticnet',
        solver='saga',
        max_iter=5000,
        random_state=42
    ))
])

# ✅ Dosya yolu ayarı
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "heart.csv")

# 🔹 Veri yükleme
df = pd.read_csv(csv_path)
X = df.drop(columns=["output"])
y = df["output"]

# 🔹 Eğit ve kaydet
full_pipeline.fit(X, y)

with open(os.path.join(BASE_DIR, "pipeline.pkl"), "wb") as f:
    pickle.dump(full_pipeline, f)
