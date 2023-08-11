from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from base_xgboost import BaseXGBoost

X, y = make_regression(n_samples=100, n_features=2, n_informative=2, noise=1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Run linear regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

y_train_pred = linear_reg.predict(X_train)

def print_predictions(X, y, y_pred):
    for i in range(len(X_train)):
        print(f'{X[i,0]:<6.3f} {X[i,1]:<6.3f} {y[i]:<8.3f} {y_pred[i]:<8.3f}')

print('Linear regression')
print_predictions(X_train, y_train, y_train_pred)

xgboost = BaseXGBoost(n_estimators=100, learning_rate=0.3, max_depth=6, reg_lambda=1)
xgboost.fit(X_train, y_train)

print('XGBoost')
y_train_pred = xgboost.predict(X_train)
print_predictions(X_train, y_train, y_train_pred)