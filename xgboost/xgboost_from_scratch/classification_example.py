import pandas as pd

from sklearn.datasets import make_classification, load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

from xgboost_classifier import XGBoostClassifier
from xgboost import XGBClassifier

#X, y = make_classification(random_state=0)
#X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, random_state=0)
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def print_predictions(X, y, y_pred):
    for i in range(len(X_train)):
        print(f'{X[i,0]:<6.3f} {X[i,1]:<6.3f} {y[i]:<8.3f} {y_pred[i]:<8.3f}')

def evaluate_model(model, X_train, y_train, X_test, y_test):
    print('-' * 35)
    train_score = model.score(X_train ,y_train)
    print(f'Accuracy (train): {train_score:.4f}')
    test_score = model.score(X_test, y_test)
    print(f'Accuracy (test): {test_score:.4f}')
    print('-' * 35)

    return train_score, test_score

print('Logistic regression')
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
evaluate_model(logistic_reg, X_train, y_train, X_test, y_test)

print('Decision tree')
tree_clf = DecisionTreeClassifier(random_state=0)
tree_clf.fit(X_train, y_train)
evaluate_model(tree_clf, X_train, y_train, X_test, y_test)

print('Gradient boosting')
gboost_clf = GradientBoostingClassifier(random_state=0)
gboost_clf.fit(X_train, y_train)
evaluate_model(gboost_clf, X_train, y_train, X_test, y_test)

print('Histogram-based gradient boosting')
hist_gboost_clf = HistGradientBoostingClassifier(random_state=0)
hist_gboost_clf.fit(X_train, y_train)
evaluate_model(hist_gboost_clf, X_train, y_train, X_test, y_test)

print('XGBoost (custom)')
my_xgboost_clf = XGBoostClassifier(verbose=1)
my_xgboost_clf .fit(X_train, y_train)
evaluate_model(my_xgboost_clf, X_train, y_train, X_test, y_test)

print('XGBoost (original)')
xgboost_clf = XGBClassifier()
xgboost_clf.fit(X_train, y_train)
evaluate_model(xgboost_clf, X_train, y_train, X_test, y_test)