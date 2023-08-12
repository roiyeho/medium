from xgb_node import XGBNode

class XGBTree:
    def __init__(self) -> None:
        self.root: XGBNode = None

    def build(self, X, grads, hessians, max_depth, min_samples_split, reg_lambda, gamma):
        self.root = XGBNode()
        curr_depth = 0
        self.root.build(X, grads, hessians, curr_depth, max_depth, min_samples_split, reg_lambda, gamma)

    def predict(self, x):
        return self.root.predict(x)