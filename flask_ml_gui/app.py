from flask import Flask, render_template, request
import pandas as pd
import pickle
import sys
sys.path.append("..")
import minibatch as Minibatch

app = Flask(__name__)

with open("../model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
poly = data["poly"]
selected_features = data["selected_features"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_dict = {
        "Education_Level": request.form["Education_Level"],
        "Occupation": request.form["Occupation"],
        "Location": request.form["Location"],
        "Work_Experience": float(request.form["Work_Experience"]),
        "Employment_Status": request.form["Employment_Status"],
        "Gender": request.form["Gender"]
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)

    for col in selected_features:
        if col not in input_df:
            input_df[col] = 0
    input_df = input_df[selected_features]


    input_scaled = scaler.transform(input_df)
    input_poly = poly.transform(input_scaled)

    prediction = input_poly.dot(model.theta)
    predicted_value = round(prediction[0], 2)

    return render_template("result.html", prediction=predicted_value)

if __name__ == "__main__":
        app.run(debug=True)