import numpy as np
from xgboost_base_model import XGBoostBaseModel
from sklearn.metrics import accuracy_score

class XGBoostClassifier(XGBoostBaseModel):
    def __init__(
        self, 
        n_estimators=100, 
        max_depth=6, 
        min_samples_split=2, 
        learning_rate=0.3, 
        reg_lambda=1, 
        gamma=0,
        verbose=0
    ):
        super().__init__(n_estimators, max_depth, min_samples_split, learning_rate, reg_lambda, gamma, verbose)

    def get_base_prediction(self, X, y):
        # Return the log odds 
        prob = np.sum(y == 1) / len(y)        
        return np.log(prob / (1 - prob))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def calc_gradients(self, y, y_pred_raw):  
        prob = self.sigmoid(y_pred_raw)    
        grads = prob - y
        return grads

    def calc_hessians(self, y, y_pred_raw):
        prob = self.sigmoid(y_pred_raw)
        hessians = prob * (1 - prob)
        return hessians
    
    def predict_proba(self, X):
        log_odds = self.get_raw_predictions(X)
        prob = self.sigmoid(log_odds)
        return prob
    
    def predict(self, X):
        prob = self.predict_proba(X)
        y_pred = np.where(prob > 0.5, 1, 0)
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)