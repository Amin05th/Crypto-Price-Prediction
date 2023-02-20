import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor, BayesianRidge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
from sklearn.decomposition import PCA


def evaluate(y_true, y_pred):
    mqe = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    try:
        rmsle = mean_squared_log_error(y_true, y_pred, squared=False)
    except Exception:
        rmsle = np.nan
    return mqe, mae, msle, rmse, rmsle, r2,


df = pd.read_csv("crypto_data.csv", index_col="Date", parse_dates=True)
X = df.drop(columns="Close")
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
preprocessed_X_train = scaler.fit_transform(X_train)
preprocessed_X_test = scaler.fit_transform(X_test)

# define models
sgd_param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1], "epsilon": [0.1, 0.4, 0.8], "eta0": [0.1, 0.4, 0.8],
                  "penalty": ["l1", "l2", "elasticnet"]}
elastic_net_param_grid = {'alpha': [0.1, 1, 10, 0.01], 'l1_ratio': np.arange(0.40, 1.00, 0.10), 'tol': [0.0001, 0.001]}

model_list = {
    "linear_regression": LinearRegression(),
    "RandomForestRegressor": GridSearchCV(RandomForestRegressor(), {"n_estimators": [200, 250, 300]}),
    "KernelRidge": KernelRidge(alpha=1.0),
    "BayesianRidge": BayesianRidge(),
    "ElasticNet": GridSearchCV(ElasticNet(max_iter=10000), param_grid=elastic_net_param_grid),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "SGDRegressor": GridSearchCV(SGDRegressor(max_iter=10000), param_grid=sgd_param_grid)
}

# getting score
model_scores = {}
for model in model_list:
    model_list[model].fit(preprocessed_X_train, y_train)
    prediction = model_list[model].predict(preprocessed_X_test)
    model_scores[model] = evaluate(y_test, prediction)

model_scores_df = pd.DataFrame(model_scores).to_csv("model_scores.csv")

# combine different models
model_list.pop("SGDRegressor")
voting_reg = VotingRegressor(estimators=list(model_list.items()))
voting_reg.fit(X_train, y_train)
y_pred = voting_reg.predict(X_test)


