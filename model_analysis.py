import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minibatch import MiniBatch
from data_cleaning import Data_cleaning
from hyperparameter_tuning import Hyperparameter_tuning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
hyperparameter_tuning = Hyperparameter_tuning()
data_cleaning = Data_cleaning()

# load data
data = pd.read_csv("data.csv")

# set seed
np.random.seed(60)

# remove data with outliers 
cleaned_data = data_cleaning.remove_outliers(data, ["Income", "Work_Experience"])

# feature selection
selected_features = ["Education_Level", "Occupation", "Location", "Work_Experience", "Employment_Status", "Gender"]

X = cleaned_data[selected_features]
y = cleaned_data["Income"].values

X = pd.get_dummies(X, drop_first=True)

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# intercept 
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# hyperparameter tuning
best_learning_rate, best_regularization_parameter, best_epoch = hyperparameter_tuning.get_best_param(X_train, X_test, y_train, y_test)

# call MBGD batch size 32
mb_model_size32 = MiniBatch()
mb_loss_size32 = mb_model_size32.fit(X_train, y_train, lr=best_learning_rate, epochs=best_epoch, batch_size=32, regularization_parameter=best_regularization_parameter)

# call MBGD2 batch size 64
mb_model_size64 = MiniBatch()
mb_loss_size64 = mb_model_size64.fit(X_train, y_train, lr=best_learning_rate, epochs=best_epoch, batch_size=64, regularization_parameter=best_regularization_parameter)

# call MBGD2 batch size 128
mb_model_size128 = MiniBatch()
mb_loss_size128 = mb_model_size128.fit(X_train, y_train, lr=best_learning_rate, epochs=best_epoch, batch_size=128, regularization_parameter=best_regularization_parameter)

# call FBGD
fb_model = MiniBatch()
fb_loss = fb_model.fit(X_train, y_train, lr=best_learning_rate, epochs=best_epoch, batch_size=None, regularization_parameter=best_regularization_parameter)

# plot MSE for epochs
plt.figure()
plt.plot(mb_loss_size32, label="Mini-Batch (batch=32)")
plt.plot(mb_loss_size64, label="Mini-Batch (batch=64)")
plt.plot(mb_loss_size128, label="Mini-Batch (batch=128)")
plt.plot(fb_loss, label="Full-Batch")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("MBGD(32) vs MBGD(64) vs MBGD(128) vs FBGD Loss Curve")
plt.legend()
plt.savefig("mbgd_vs_fbgd_loss_curve.png")

# predict 
mb_pred_size32 = X_test.dot(mb_model_size32.theta)
mb_pred_size64 = X_test.dot(mb_model_size64.theta)
mb_pred_size128 = X_test.dot(mb_model_size128.theta)
fb_pred = X_test.dot(fb_model.theta)

# calculate MSE
mb_mse_size32 = mean_squared_error(y_test, mb_pred_size32)
mb_mse_size64 = mean_squared_error(y_test, mb_pred_size64)
mb_mse_size128 = mean_squared_error(y_test, mb_pred_size128)
fb_mse = mean_squared_error(y_test, fb_pred)

# MSE comparison
plt.figure()
plt.bar(["Mini-Batch (32)", "Mini-Batch (64)", "Mini-Batch (128)", "Full-Batch"], [mb_mse_size32, mb_mse_size64, mb_mse_size128, fb_mse])
plt.ylabel("Test MSE")
plt.title("MBGD(32) vs MBGD(64) vs MBGD(128) vs FBGD: Final Test MSE")
plt.savefig("mbgd_vs_fbgd_test_mse.png")