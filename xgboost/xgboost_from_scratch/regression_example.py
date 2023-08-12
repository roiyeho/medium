import pandas as pd

from sklearn.datasets import make_regression
from sklearn.datasets import fetch_california_housing, load_diabetes

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE

from xgboost import XGBRegressor
from xgboost_regressor import XGBoostRegressor

#X, y = make_regression(n_samples=1000, n_features=2, n_informative=2, noise=5, random_state=0)
#X, y = make_regression(noise=1, random_state=0)

#data = fetch_california_housing()
#X, y = data.data, data.target

df_train = pd.read_csv('./test_data/regression.train', header=None, sep='\t')
df_test = pd.read_csv('./test_data/regression.test', header=None, sep='\t')
X_train = df_train.drop(0, axis=1).values
X_test = df_test.drop(0, axis=1).values
y_train = df_train[0].values
y_test = df_test[0].values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def print_predictions(X, y, y_pred):
    for i in range(len(X_train)):
        print(f'{X[i,0]:<6.3f} {X[i,1]:<6.3f} {y[i]:<8.3f} {y_pred[i]:<8.3f}')

def evaluate_model(model, X_train, y_train, X_test, y_test):
    print('-' * 35)
    train_score = model.score(X_train ,y_train)
    print(f'R2 score (train): {train_score:.4f}')
    test_score = model.score(X_test, y_test)
    print(f'R2 score (test): {test_score:.4f}')

    y_train_pred = model.predict(X_train)
    train_rmse = MSE(y_train, y_train_pred, squared=False)
    print(f'RMSE (train): {train_rmse:.4f}')
    y_test_pred = model.predict(X_test)
    test_rmse = MSE(y_test, y_test_pred, squared=False)
    print(f'RMSE (test): {test_rmse:.4f}')
    print('-' * 35)

    return train_score, test_score

print('Linear regression')
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
evaluate_model(linear_reg, X_train, y_train, X_test, y_test)

print('Decision tree')
tree_reg = DecisionTreeRegressor(random_state=0)
tree_reg.fit(X_train, y_train)
evaluate_model(tree_reg, X_train, y_train, X_test, y_test)

print('Gradient boosting')
gboost_reg = GradientBoostingRegressor(n_estimators=30, random_state=0)
gboost_reg.fit(X_train, y_train)
evaluate_model(gboost_reg, X_train, y_train, X_test, y_test)

print('Histogram-based gradient boosting')
hist_gboost_reg = HistGradientBoostingRegressor(max_iter=30, random_state=0)
hist_gboost_reg.fit(X_train, y_train)
evaluate_model(hist_gboost_reg, X_train, y_train, X_test, y_test)

print('XGBoost (original)')
xgboost_reg = XGBRegressor(n_estimators=30)
xgboost_reg.fit(X_train, y_train)
evaluate_model(xgboost_reg, X_train, y_train, X_test, y_test)

print('XGBoost (custom)')
my_xgboost_reg = XGBoostRegressor(n_estimators=30, learning_rate=0.3, max_depth=6, reg_lambda=1, verbose=1)
my_xgboost_reg.fit(X_train, y_train)
evaluate_model(my_xgboost_reg, X_train, y_train, X_test, y_test)