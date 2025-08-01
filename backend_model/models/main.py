import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# 🔹 Özellik mühendisliği class'ı
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["yas_grubu"] = pd.cut(X["age"], bins=[0,40,55,70,100], labels=["<40", "40-55", "56-70", ">70"])
        X["oldpeak_yuksek"] = (X["oldpeak"] > 2.0).astype(int)
        X["egim_flat_veya_down"] = X["slp"].isin([1, 2]).astype(int)
        X["sessiz_gogus_agrisi"] = (X["cp"] == 4).astype(int)
        X["thal_geri_donen_defekt"] = (X["thall"] == 2).astype(int)
        X["exng_oldpeak_carpim"] = X["exng"] * X["oldpeak"]
        X["thalach_age_orani"] = X["thalachh"] / X["age"]
        X["risk_skoru_light"] = X["age"] + X["trtbps"] + X["chol"] + X["oldpeak"] - X["thalachh"]
        return X

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

# 🔹 Veri yükleme
df = pd.read_csv("heart.csv")
X = df.drop(columns=["output"])
y = df["output"]

# 🔹 Eğit ve kaydet
full_pipeline.fit(X, y)

with open("pipeline.pkl", "wb") as f:
    pickle.dump(full_pipeline, f)
