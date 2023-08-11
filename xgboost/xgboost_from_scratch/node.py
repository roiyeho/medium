import numpy as np

class Node:
    """A node object that recursively builds itself to construct a regression tree.
    """
    def __init__(self) -> None:
        self.is_leaf: bool = False
        self.left_child: Node = None
        self.right_child: Node = None
        self.split_feature_idx: int = None
        self.split_threshold: float = None
        self.weight: float = None

    def build(self, X, grads, hessians, curr_depth, max_depth, min_samples_split, reg_lambda, gamma):
        """Exact greedy algorithm for split finding
        """
        if len(X) < min_samples_split or curr_depth >= max_depth:
            # Stopping criterion (1): there are less than min_samples_split samples at the node
            # Stopping criterion (2): max depth of the tree has been reached 
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grads, hessians, reg_lambda)
            return
        
        best_gain, best_feature_idx, best_threshold, best_left_samples_idx, best_right_samples_idx = \
            self._find_best_split(X, grads, hessians, reg_lambda)
                   
        if best_gain < gamma:
            # Stopping criterion (3): the best gain is less than the minimum split gain 
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grads, hessians, reg_lambda)
            return        
        else:
            # Split the node according to the best split found
            self.split_feature_idx = best_feature_idx
            self.split_threshold = best_threshold

            self.left_child = Node()
            self.left_child.build(X[best_left_samples_idx],
                                grads[best_left_samples_idx],
                                hessians[best_left_samples_idx],
                                curr_depth + 1,
                                max_depth, min_samples_split, reg_lambda, gamma)
            
            self.right_child = Node()
            self.right_child.build(X[best_right_samples_idx],
                                grads[best_right_samples_idx],
                                hessians[best_right_samples_idx],
                                curr_depth + 1,
                                max_depth, min_samples_split, reg_lambda, gamma)
       
    def _find_best_split(self, X, grads, hessians, reg_lambda):
        """Scans through every feature and find the best split point. 
        """
        G = np.sum(grads)
        H = np.sum(hessians)

        best_gain = float('-inf')   
        best_feature_idx = None
        best_threshold = None
        best_left_samples_idx = None
        best_right_samples_idx = None
        
        # Check all possible features
        for j in range(X.shape[1]):
            G_left, H_left = 0, 0
            sorted_samples_idx = np.argsort(X[:, j])

            # For each feature calculate the gain at every possible split
            for i in range(X.shape[0] - 1):   
                G_left += grads[sorted_samples_idx[i]]
                H_left += hessians[sorted_samples_idx[i]]

                G_right = G - G_left
                H_right = H - H_left
                curr_gain = self._calc_split_gain(G, H, G_left, H_left, G_right, H_right, reg_lambda)

                if curr_gain > best_gain:
                    best_gain = curr_gain                    
                    best_feature_idx = j          
                    best_threshold = X[sorted_samples_idx[i]][j]
                    best_left_samples_idx = sorted_samples_idx[:i + 1]
                    best_right_samples_idx = sorted_samples_idx[i + 1:]

        return best_gain, best_feature_idx, best_threshold, best_left_samples_idx, best_right_samples_idx

    def _calc_leaf_weight(self, grads, hessians, reg_lambda):
        """Calculate the optimal weight of this leaf node
        """
        return -np.sum(grads) / (np.sum(hessians) + reg_lambda)
    
    def _calc_split_gain(self, G, H, G_left, H_left, G_right, H_right, reg_lambda):
        """Compute the loss reduction
        """
        def calc_term(g, h):
            return g**2 / (h + reg_lambda)

        gain = calc_term(G_left, H_left) + calc_term(G_right, H_right) - calc_term(G, H)
        return 0.5 * gain
        
    def predict(self, x):
        """Predict the label for a new sample x
        """
        if self.is_leaf:
            return self.weight
        else:
            if x[self.split_feature_idx] <= self.split_threshold:
                return self.left_child.predict(x)
            else:
                return self.right_child.predict(x)  