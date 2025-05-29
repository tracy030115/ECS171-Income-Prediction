import numpy as np
from sklearn.metrics import mean_squared_error

#usage:
#model = MiniBatch()
#model.fit(X_train_scaled, y_train, lr = <learning rate>, epochs = <epochs>, batch_size = <batch size>)

class MiniBatch:
    def __init__(self):
        self.theta = None
    
    def fit(self, X, y, lr=0.01, epochs=20, batch_size=None, verbose=True, regularization_parameter=0.01):
        self.theta = np.random.randn(X.shape[1])
        m = X.shape[0]
        loss_history = []

        for epoch in range(epochs):
            num_batches = m // batch_size if batch_size else 1
            for batch in range(num_batches):
                if batch_size:
                    indices = np.random.choice(m, batch_size)
                    X_batch = X[indices]
                    y_batch = y[indices]
                else:
                    X_batch = X
                    y_batch = y
                y_pred = X_batch.dot(self.theta)
                error = y_pred - y_batch
            
                gradient = 2 * X_batch.T.dot(error) + 2 * regularization_parameter * self.theta
                avg_gradient = sum(gradient)/ X_batch.shape[0]
            
                self.theta -= lr * avg_gradient

            y_train_pred = X.dot(self.theta)
            mse = mean_squared_error(y, y_train_pred)
            l2_penalty = regularization_parameter * np.sum(self.theta ** 2)
            loss_history.append(mse + l2_penalty)
        return loss_history
