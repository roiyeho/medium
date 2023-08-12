import numpy as np
from xgboost_base_model import XGBoostBaseModel
from sklearn.metrics import r2_score

class XGBoostRegressor(XGBoostBaseModel):
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
        return np.mean(y)
    
    def calc_gradients(self, y, y_pred_raw):        
        grads = 2 * (y_pred_raw - y)
        return grads

    def calc_hessians(self, y, y_pred_raw):
        hessians = np.full(len(y), 2)
        return hessians
    
    def predict(self, X):
        y_pred = self.get_raw_predictions(X)
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)