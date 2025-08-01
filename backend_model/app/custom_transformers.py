import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
