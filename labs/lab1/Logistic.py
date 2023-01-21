import numpy as np
class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        self.weights = np.zeros((12, 1))
        self.train_loss = []
        self.penalty = penalty
        err_msg = "vpenalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        return 1 / (1 + (np.exp(-x)))

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
        """
        Fit the regression coefficients via gradient descent or other methods 
        """
        y = np.expand_dims(y, axis=1)

        for iteration in range(int(max_iter)):
            p = self.sigmoid(np.dot(X, self.weights))
            if self.penalty == 'l2':
                grad = - np.dot(X.T, y - p) + self.weights
            elif self.penalty == 'l1':
                l1 = np.ones_like(self.weights)
                l1[np.where(self.weights < 0)] = 0
                grad = - np.dot(X.T, y - p) + l1

            los = (-np.dot(np.dot(y.T, X), self.weights) + np.sum(np.log(1 + np.exp(np.dot(X, self.weights)))))
            self.train_loss.append(los[0][0])

            if (np.absolute(grad) < tol).all():
                print(iteration)
                break
            
            self.weights = self.weights - lr * grad
        
    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        pred = self.sigmoid(np.dot(X, self.weights))
        return pred
