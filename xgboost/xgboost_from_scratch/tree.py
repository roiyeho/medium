from node import Node

class Tree:
    def __init__(self) -> None:
        self.root: Node = None

    def build(self, X, grads, hessians, max_depth, min_samples_split, reg_lambda, gamma):
        self.root = Node()
        curr_depth = 0
        self.root.build(X, grads, hessians, curr_depth, max_depth, min_samples_split, reg_lambda, gamma)

    def predict(self, x):
        return self.root.predict(x)