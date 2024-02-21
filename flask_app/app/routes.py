# src/app/routes.py
from flask import Flask, render_template, request
import joblib
from sklearn import datasets
import numpy as np
import pickle
import pkgutil

app = Flask(__name__)

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
feature_names = iris.feature_names

# Load the saved logistic regression model
with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


@app.route("/")
def index():
    return render_template("index.html", feature_names=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the request
        features = [float(request.form.get(feature)) for feature in feature_names]
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template(
            "index.html", feature_names=feature_names, prediction=prediction
        )

    except Exception as e:
        return render_template("index.html", feature_names=feature_names, error=str(e))
