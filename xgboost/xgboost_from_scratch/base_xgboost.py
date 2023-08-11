import numpy as np

from sklearn.base import BaseEstimator
from tree import Tree
from typing import List

class BaseXGBoost(BaseEstimator):
    def __init__(
        self,
        n_estimators=1,             # number of boosting rounds
        max_depth=2,                # Maximum depth of a tree
        min_samples_split=2,        # The minimum number of samples required to split an internal node
        learning_rate=0.3,          # Step size shrinkage applied to the leaf weights
        reg_lambda=1,               # L2 regularization term on the leaf weights
        gamma=0                     # Minimum loss reduction to make a split on a leaf node
    ):
        self.n_estimators = n_estimators        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        
    def fit(self, X, y):
        self.base_pred = self._get_base_prediction(X, y)
        self.estimators: List[Tree] = []

        for i in range(self.n_estimators):
            y_pred = self.predict(X)
            grads = self._calc_gradients(y, y_pred)
            hessians = self._calc_hessians(y, y_pred)

            tree = Tree()
            tree.build(X, grads, hessians, self.max_depth, self.min_samples_split, self.reg_lambda, self.gamma)
            self.estimators.append(tree)
            if i % 10 == 0:
                print(f'Boosting iteration {i}')

        print('Training finished')
        return self
    
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_pred)
        if len(self.estimators) > 0:
            for i in range(len(X)):            
                y_pred[i] += np.sum(self.learning_rate * estimator.predict(X[i]) 
                                    for estimator in self.estimators)
        return y_pred
    
    def _get_base_prediction(self, X, y):
        return np.mean(y)
    
    def _calc_gradients(self, y, y_pred):        
        grads = 2 * (y_pred - y)
        return grads

    def _calc_hessians(self, y, y_pred):
        hessians = np.full(len(y), 2)
        return hessians