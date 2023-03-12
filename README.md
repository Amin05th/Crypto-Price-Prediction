# Crypto-Price-Prediction: Project Overview

- Created an app that predictes crypto currency to help people to inversting in BTC
- Downloaded Dataset from pandas_datareader
- Optimized VotingRegressor using LinearRegression, SGDRegressor, BayesianRidge, ElasticNet, RandomForestRegressor, GradientBoostingRegressor and KernelRidge
- Built a Userinterface with streamlit

# Code and Resources Used

- **Python Version**: 3.10
- **Packages**: numpy, pandas, pandas_datareader, sklearn, matplotlib, plotly, seaborn, pickle, datetime and streamlit
- **Data Resource**: pandas_datareader currency: BTC, USD, data: 2016-2023

# Pandas_datareader Data

For each day, we got the following features:

- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

# EDA

- created line diagram
- created animated line diagram

![Your Browser does not support Images](./Dokumente/Python/Crypto Price Prediction/Figure_1.png)

# Model Building

First I scaled Data using MinMaxScaler then I split them using train_test_split with test_size equal to 20%

I tried for evaluation porpuses  mean_squared_error, mean_absolute_error, mean_squared_log_error and r2_score

I tried Votingregressor using 7 different Regresson Models:
- LinearRegression
- SGDRegressor
- BayesianRidge
- ElasticNet
- RandomForestRegressor
- GradientBoostingRegressor
- KernelRidge

# Productionization

In this step I built a streamlit app to predict using different parameter like how many years in future do you want to predict and plotted a dynamic Plot
