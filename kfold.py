import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def kf_minibatch(X, y, model_class, folds=5, lr=0.01, epochs=100, batch_size=32, random_state=None):
    # Define number of folds for cross-validation
    kf = KFold(folds)

    # Initialize lists to store results for variance, bias2s, total_error, models, r2, and mse
    variance = []
    bias2 = []
    total_error = []
    models = []
    r2 = []
    mse = []

    # Perform cross-validation
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        # Split data into training and testing sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit polynomial regression model
        model = model_class()
        model.fit(X_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size)

        # Make predictions on the test set
        y_pred = X_test.dot(model.theta)

        # Calculate variance and bias for this fold
        variance_fold = np.var(y_pred)
        bias2_fold = np.mean((np.mean(y_pred) - y_test) ** 2) #𝐵𝑖𝑎𝑠2 = 𝐸[( 𝐸[𝑔(𝑥)] − 𝑓(𝑥) )^2 ]
        total_error_fold = variance_fold + bias2_fold #𝐸𝑟𝑟𝑜𝑟 𝑜𝑓 𝑡ℎ𝑒 𝑚𝑜𝑑𝑒𝑙 = 𝐵𝑖𝑎𝑠2 + 𝑉𝑎𝑟𝑖𝑎𝑛𝑐𝑒 + 𝐼𝑟𝑟𝑒𝑑𝑢𝑐𝑖𝑏𝑙𝑒 𝐸𝑟𝑟𝑜𝑟

        # Calculate R2 and MSE for this fold
        r2_fold = r2_score(y_test, y_pred)
        mse_fold = mean_squared_error(y_test, y_pred)

        # Append results to lists
        variance.append(variance_fold)
        bias2.append(bias2_fold)
        total_error.append(total_error_fold)
        models.append(model)
        r2.append(r2_fold)
        mse.append(mse_fold)

        # Print results for this fold
        print("Variance: {:.4f}, Bias2: {:.4f}, Total error: {:.4f}, R^2: {:.4f}, MSE: {:.4f}".format(variance_fold, bias2_fold, total_error_fold, r2_fold, mse_fold))

    # print the total_error of the best model
    min_error_index = np.argmin(total_error)
    best_model = models[min_error_index]
    print("Total error of the best model: {:.4f}".format(total_error[min_error_index]))

    return variance, bias2, total_error, best_model, r2, mse

# Example Usage

# Stochastic Gradient Descent
#k_fold_validate_minibatch(X, y, MiniBatch, batch_size=1)

# Full Batch Gradient Descent
#k_fold_validate_minibatch(X, y, MiniBatch, batch_size=None)

# Mini-Batch
#k_fold_validate_minibatch(X, y, MiniBatch, batch_size=x)
