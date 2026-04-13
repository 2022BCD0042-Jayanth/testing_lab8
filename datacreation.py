import pandas as pd

df = pd.read_csv("housing_full.csv")
df.head(5000).to_csv("data/housing.csv", index=False)