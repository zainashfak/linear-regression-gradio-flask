import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train(csv_path, target, name):
    df = pd.read_csv(csv_path).dropna()

    X = df.drop(columns=[target])          # input features
    y = df[target]                         # value we predict
    X = pd.get_dummies(X, drop_first=True) # text columns -> 0/1 columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("\n", name)
    print("R2  :", r2_score(y_test, pred))
    print("MAE :", mean_absolute_error(y_test, pred))
    print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)

    joblib.dump({"model": model, "columns": list(X.columns)}, f"models/{name}.pkl")


train("data/insurance.csv", "charges", "insurance")
train("data/concrete.csv", "strength", "concrete")