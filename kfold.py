import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression(degree, X, y, folds, test_size=0.25, random_state=None):
    # Define number of folds for cross-validation
    kf = KFold(folds)

    # Initialize lists to store results for variance, bias2s, total_error, models, r2, and mse
    variance = []
    bias2 = []
    total_error = []
    models = []
    r2 = []
    mse = []

    # Set the polynomial degree of the model
    poly_features = PolynomialFeatures(degree)
    X_poly = poly_features.fit_transform(X)

    # Perform cross-validation
    for train_index, test_index in kf.split(X_poly):
        # Split data into training and testing sets for this fold
        X_train, X_test = X_poly[train_index], X_poly[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit polynomial regression model
        model = MiniBatch()
        model.fit(X_train, y_train)

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

        # Print results for this fold
        print("Variance: {:.4f}, Bias2: {:.4f}, Total error: {:.4f}, R^2: {:.4f}, MSE: {:.4f}".format(variance_fold, bias2_fold, total_error_fold, r2_fold, mse_fold))

    # print the total_error of the best model
    min_error_index = np.argmin(total_error)
    best_model = models[min_error_index]
    print("Total error of the best model: {:.4f}".format(total_error[min_error_index]))

    # Testing the final model on the test data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=42)
    y_pred = best_model.predict(X_test)

    # Store mse and r2 score of the model applied on the test data
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print("Test MSE: {:.4f}".format(test_mse))
    print("Test R^2: {:.4f}".format(test_r2))

    return test_mse, best_model, total_error[min_error_index] # Can be modified to return whatever we need
