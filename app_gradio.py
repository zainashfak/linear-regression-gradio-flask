import gradio as gr
import joblib
import pandas as pd

# ---------- Insurance model ----------
ins = joblib.load("models/insurance.pkl")


def predict_insurance(age, sex, bmi, children, smoker, region):
    row = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi,
                         "children": children, "smoker": smoker, "region": region}])
    row = pd.get_dummies(row).reindex(columns=ins["columns"], fill_value=0)
    return round(float(ins["model"].predict(row)[0]), 2)


ui_insurance = gr.Interface(
    fn=predict_insurance,
    inputs=[
        gr.Number(label="Age", value=30),
        gr.Dropdown(["male", "female"], label="Sex", value="male"),
        gr.Number(label="BMI", value=25),
        gr.Number(label="Children", value=0),
        gr.Dropdown(["yes", "no"], label="Smoker", value="no"),
        gr.Dropdown(["southwest", "southeast", "northwest", "northeast"],
                    label="Region", value="southwest"),
    ],
    outputs=gr.Number(label="Predicted charges"),
    title="Insurance Cost Predictor",
)

# ---------- Concrete model ----------
con = joblib.load("models/concrete.pkl")


def predict_concrete(cement, slag, ash, water, superplastic, coarseagg, fineagg, age):
    row = pd.DataFrame([{"cement": cement, "slag": slag, "ash": ash, "water": water,
                         "superplastic": superplastic, "coarseagg": coarseagg,
                         "fineagg": fineagg, "age": age}])
    row = row.reindex(columns=con["columns"], fill_value=0)
    return round(float(con["model"].predict(row)[0]), 2)


ui_concrete = gr.Interface(
    fn=predict_concrete,
    inputs=[
        gr.Number(label="Cement", value=300),
        gr.Number(label="Slag", value=0),
        gr.Number(label="Ash", value=0),
        gr.Number(label="Water", value=180),
        gr.Number(label="Superplasticizer", value=5),
        gr.Number(label="Coarse aggregate", value=1000),
        gr.Number(label="Fine aggregate", value=750),
        gr.Number(label="Age (days)", value=28),
    ],
    outputs=gr.Number(label="Predicted strength (MPa)"),
    title="Concrete Strength Predictor",
)

# ---------- Combine into tabs ----------
demo = gr.TabbedInterface([ui_insurance, ui_concrete], ["Insurance", "Concrete"])

if __name__ == "__main__":
    demo.launch()