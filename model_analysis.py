import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minibatch import MiniBatch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# load data
data = pd.read_csv("data.csv")

# get features for income
X = pd.get_dummies(data.drop(columns=["Income"]), drop_first=True)
y = data["Income"].values

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# intercept 
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# call MBGD
mb_model = MiniBatch()
mb_loss = mb_model.fit(X_train, y_train, lr=0.01, epochs=50, batch_size=32)

# call FBGD
fb_model = MiniBatch()
fb_loss = fb_model.fit(X_train, y_train, lr=0.01, epochs=50, batch_size=None)

# plot MSE for epochs
plt.figure()
plt.plot(mb_loss, label="Mini-Batch (batch=32)")
plt.plot(fb_loss, label="Full-Batch")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("MBGD vs FBGD Loss Curve")
plt.legend()
plt.savefig("mbgd_vs_fbgd_loss_curve.png")

# predict 
mb_pred = X_test.dot(mb_model.theta)
fb_pred = X_test.dot(fb_model.theta)

# calculate MSE
mb_mse = mean_squared_error(y_test, mb_pred)
fb_mse = mean_squared_error(y_test, fb_pred)

# MSE comparison
plt.figure()
plt.bar(["Mini-Batch", "Full-Batch"], [mb_mse, fb_mse])
plt.ylabel("Test MSE")
plt.title("MBGD vs FBGD: Final Test MSE")
plt.savefig("mbgd_vs_fbgd_test_mse.png")