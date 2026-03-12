import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_forecast_sarimax(forecast, X_train, X_test, confidence_int):
    plt.figure(figsize=(12,6))
    plt.plot(X_train, label="train")
    plt.plot(X_test, label="test")
    plt.plot(forecast, label="forecast")

    plt.fill_between(
    confidence_int.index,
    confidence_int.iloc[:,0],
    confidence_int.iloc[:,1],
    alpha=0.7)

    plt.legend()
    plt.show()

def plot_forecast_xgboost(y_train, y_val, y_test, y_val_pred, y_test_pred):

    # params = [y_train, y_val, y_test, y_val_pred, y_val_test]

    # for param in params:
    #     f'{param}_day' = f'{param}'.resample('D').mean()

    index_val = y_val.index
    index_test = y_test.index

    y_val_pred = pd.Series(y_val_pred, index=index_val)
    y_test_pred = pd.Series(y_test_pred, index=index_test)

    y_train_day = y_train.resample('D').mean()
    y_val_day = y_val.resample('D').mean()
    y_test_day = y_test.resample('D').mean()
    y_val_pred_day = y_val_pred.resample('D').mean()
    y_val_test_day = y_test_pred.resample('D').mean()


    plt.figure(figsize=(12,6))
    plt.plot(y_train_day, label="train")
    plt.plot(y_test_day, label="test")
    plt.plot(y_val_pred_day, label="val_pred")
    plt.plot(y_val_test_day, label="test_pred")
