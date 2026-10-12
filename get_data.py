import pandas as pd

base = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/"
pd.read_csv(base + "insurance.csv").to_csv("data/insurance.csv", index=False)
pd.read_csv(base + "concrete.csv").to_csv("data/concrete.csv", index=False)
print("done")