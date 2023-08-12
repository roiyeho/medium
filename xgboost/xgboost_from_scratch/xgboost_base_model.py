import time
import numpy as np

from sklearn.base import BaseEstimator
from xgb_tree import XGBTree
from typing import List
from abc import ABC, abstractmethod

class XGBoostBaseModel(ABC, BaseEstimator):
    def __init__(
        self,
        n_estimators=100,       # The number of trees (boosting rounds)
        max_depth=6,            # Maximum depth of a tree
        min_samples_split=2,    # Minimum number of samples required to split an internal node
        learning_rate=0.3,      # Step size shrinkage applied to the leaf weights
        reg_lambda=1,           # L2 regularization term on the leaf weights
        gamma=0,                # Minimum loss reduction to make a split on a leaf node
        verbose=0               # Controls the verbosity of the log message (0 or 1)
    ):
        self.n_estimators = n_estimators        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.verbose = verbose
        
    def fit(self, X, y):
        train_start_time = time.time()
        self.base_pred = self.get_base_prediction(X, y)
        self.estimators: List[XGBTree] = []

        for i in range(self.n_estimators):
            y_pred_raw = self.get_raw_predictions(X)
            grads = self.calc_gradients(y, y_pred_raw)
            hessians = self.calc_hessians(y, y_pred_raw)

            tree = XGBTree()
            tree.build(X, grads, hessians, self.max_depth, self.min_samples_split, self.reg_lambda, self.gamma)
            self.estimators.append(tree)
            if self.verbose and i % 10 == 0:
                print(f'Boosting iteration {i}')

        if self.verbose:
            elapsed = time.time() - train_start_time
            print(f'Training finished. Time elapsed: {elapsed:.3f} sec')
        return self
    
    def get_raw_predictions(self, X):
        y_pred_raw = np.full(X.shape[0], self.base_pred)
        if len(self.estimators) > 0:
            for i in range(len(X)):            
                y_pred_raw[i] += np.sum(self.learning_rate * estimator.predict(X[i]) 
                                        for estimator in self.estimators)
        return y_pred_raw
    
    @abstractmethod
    def get_base_prediction(self, X, y):
        pass
    
    @abstractmethod
    def calc_gradients(self, y, y_pred_raw):        
        pass

    @abstractmethod
    def calc_hessians(self, y, y_pred_raw):
        pass

    @abstractmethod
    def predict(self, X):
        pass