import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import joblib
import os

# Load data
df = pd.read_csv("data/housing.csv")

# Split
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

# Save metrics
os.makedirs("metrics", exist_ok=True)
metrics = {
    "rmse": rmse,
    "r2": r2,
    "dataset_size": len(df)
}

with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print(metrics)
