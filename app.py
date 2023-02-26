import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime as dt, timedelta
import pickle
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


def load_model():
    with open("model_file.p", "rb") as pickled:
        mod = pickle.load(pickled)
        mod = mod['model']
    return mod


def make_future_dataframe(mod, mor, df_for_pred, df_to_pred):
    for index in df_to_pred.index.values:
        row_to_prediction = pd.DataFrame(df_to_pred.loc[index]).T
        X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(df_for_pred.drop(columns="Close"),
                                                                            df_for_pred["Close"],
                                                                            test_size=0.30, random_state=0, shuffle=False)

        mod.fit(X_train_mod, y_train_mod)
        close_prediction = mod.predict(X_test_mod).mean()
        row_to_prediction["Close"] = close_prediction

        X_train_mor, X_test_mor, y_train_mor, y_test_mor = train_test_split(df_for_pred["Close"],
                                                                            df_for_pred.drop(columns="Close"),
                                                                            test_size=0.30, random_state=0, shuffle=False)
        mor.fit(X_train_mor.values.reshape(-1, 1), y_train_mor)
        y_pred = np.mean(mor.predict(X_test_mor.values.reshape(-1, 1)), axis=0)
        row_to_prediction[["Open", "High", "Low", "Adj Close", "Volume"]] = y_pred
        df_for_pred = pd.concat([df_for_pred, row_to_prediction])
    return df_for_pred


df = pd.read_csv("crypto_data.csv", index_col="Date", parse_dates=True)
df["Volume"] = df["Volume"].astype(float)
st.set_page_config(page_title="BTC-USD Prediction App")
st.title("BTC-USD Prediction App")

days = st.slider("number of months for forcast", 0, 5)
period = days * 30
model = RandomForestRegressor() # model_load()
mor = MultiOutputRegressor(RandomForestRegressor())

future_dates = pd.date_range(start="2023-02-21", end=dt.now() + timedelta(days=period), freq="D")
future_df = pd.DataFrame(index=future_dates)

data = pd.concat([df, future_df])

data_for_predictions = data[data.notna().any(axis=1)]
data_to_predict = data[data.isna().any(axis=1)]

future_data = make_future_dataframe(model, mor, data_for_predictions, data_to_predict)

fig = go.Figure()
fig.add_trace(go.Scatter(x=future_data.index, y=future_data["Close"]))
st.plotly_chart(fig)
