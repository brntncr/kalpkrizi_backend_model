import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/heart.csv")
print(df.columns)

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Dataset
df = pd.read_csv("data/raw/heart.csv")

# Sütun listeleri
numeric_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']

# One-hot encoding (drop_first=True)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# X, y ayır
X = df_encoded.drop(columns=["output"], axis=1)
y = df_encoded["output"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sadece numeric sütunları scale et (önce one-hot sonrası sütun isimlerine bakalım)
numeric_features = [col for col in X_train.columns if any(n in col for n in numeric_cols)]
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Kaydet
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Ayrıca sütun sırasını da kaydedelim (API prediction için gerekli!)
with open("models/feature_order.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)