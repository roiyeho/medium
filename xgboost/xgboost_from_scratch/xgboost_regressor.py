from base_xgboost import BaseXGBoost

class XGBoostClassifier(BaseXGBoost):
    def __init__(self, n_estimators=100, max_depth=2, min_samples_split=2, learning_rate=0.3, reg_lambda=1, gamma=0):
        super().__init__(n_estimators, max_depth, min_samples_split, learning_rate, reg_lambda, gamma)

        