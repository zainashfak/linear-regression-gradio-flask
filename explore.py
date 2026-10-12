import pandas as pd

for f in ["insurance", "concrete"]:
    df = pd.read_csv(f"data/{f}.csv")
    print("=====", f)
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())