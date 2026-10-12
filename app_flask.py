from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
ins = joblib.load("models/insurance.pkl")
con = joblib.load("models/concrete.pkl")


@app.route("/", methods=["GET", "POST"])
def insurance():
    result = None
    if request.method == "POST":
        row = pd.DataFrame([{
            "age": float(request.form["age"]),
            "sex": request.form["sex"],
            "bmi": float(request.form["bmi"]),
            "children": int(request.form["children"]),
            "smoker": request.form["smoker"],
            "region": request.form["region"],
        }])
        row = pd.get_dummies(row).reindex(columns=ins["columns"], fill_value=0)
        result = round(float(ins["model"].predict(row)[0]), 2)
    return render_template("insurance.html", result=result)


@app.route("/concrete", methods=["GET", "POST"])
def concrete():
    result = None
    if request.method == "POST":
        fields = ["cement", "slag", "ash", "water",
                  "superplastic", "coarseagg", "fineagg", "age"]
        row = pd.DataFrame([{f: float(request.form[f]) for f in fields}])
        row = row.reindex(columns=con["columns"], fill_value=0)
        result = round(float(con["model"].predict(row)[0]), 2)
    return render_template("concrete.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)