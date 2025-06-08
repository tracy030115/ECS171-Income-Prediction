import pickle
import numpy as np
import pandas as pd
from minibatch import MiniBatch
from kfold import kf_minibatch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# load data
data = pd.read_csv("data.csv")

# set seed
np.random.seed(60)

# feature selection
selected_features = ["Education_Level", "Occupation", "Location", "Work_Experience", "Employment_Status", "Gender"]

# get features for income
X = data[selected_features]
y = data["Income"].values

X = pd.get_dummies(X, drop_first=True)

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

variance, bias2, total_error, best_model, r2_scores, mse_scores = kf_minibatch(X_poly, y, MiniBatch, folds=5, lr=0.00001, epochs=50, batch_size=32, random_state=42)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# call best model
mb_model = MiniBatch()
mb_loss32 = mb_model.fit(X_train, y_train, lr=0.00001, regularization_parameter=0.00001, epochs=50, batch_size=32)

# save best model
model = {
    "model": mb_model,
    "scaler": scaler,
    "poly": poly,
    "selected_features": X.columns.tolist()
}

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)