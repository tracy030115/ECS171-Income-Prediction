import itertools
from sklearn.metrics import mean_squared_error
from minibatch import MiniBatch

class Hyperparameter_tuning:

    def get_best_param(self, X_train, X_test, y_train, y_test):

        # parameters to be tuned
        param = {
            'learning_rate': [0.00005, 0.00002, 0.00001],
            'regularization_parameter': [0.00005, 0.00002, 0.00001],
            'epoch': [20, 30, 40]
        }

        best_mse = float('inf')
        best_learning_rate = 0
        best_regularization_parameter = 0
        best_epoch = 0

        # grid search to find best hyperparameters
        for learning_rate, regularization_parameter, epoch in itertools.product(param['learning_rate'], param['regularization_parameter'], param['epoch']):
            model = MiniBatch()
            model.fit(X_train, y_train, lr=learning_rate, epochs=epoch, batch_size=32, regularization_parameter=regularization_parameter)

            y_pred = X_test.dot(model.theta)
            mse = mean_squared_error(y_test, y_pred)

            # record hyperparameter with min mse
            if mse < best_mse:
                best_learning_rate = learning_rate
                best_regularization_parameter = regularization_parameter
                best_epoch = epoch
                
        # return best hyperparameters
        return best_learning_rate, best_regularization_parameter, best_epoch