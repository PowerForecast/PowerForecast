import numpy as np
import pandas as pd
from power_forecast.params import *
from power_forecast.logic.get_data.build_dataframe import (
    build_common_dataframe,
    add_features_XGB,
    add_features_RNN,
)
from power_forecast.logic.preprocessing.train_test_split import (
    train_test_split_general,
    train_test_split_RNN_optimized,
    train_test_split_XGB_optimized,
)

# from power_forecast.logic.get_data.features_engineering import build_common_dataframe, add_features_XGB

df_common = build_common_dataframe(
    filepath="raw_data/all_countries.csv",
    country_objective="France",
    target_day_distance=2,
    time_interval="h",
    keep_only_neighbors=True,
    add_meteo=True,
    add_crisis=True,
    add_entsoe=True,
)

# Define inputs
preprocess_xgb = True
preprocess_rnn = True
max_train_test_split = True
objective_day = pd.Timestamp("2024-03-20", tz="UTC")
cutoff_day = pd.Timestamp("2023-10-01", tz="UTC")


if preprocess_xgb:
    df_xgb = add_features_XGB(
        df_common,
        country_objective="France",
        target_day_distance=2,
        add_lag_frontiere=False,
        drop_initial_nans=True,
    )

    columns_xgb = df_xgb.columns
    print(df_xgb.shape)
    print(columns_xgb)

if preprocess_rnn:
    df_rnn = add_features_RNN(
        df=df_common,
        country_objective="France",
        target_day_distance=2,
        add_future_time_features=True,
        add_future_meteo=True,
    )

    columns_rnn = df_rnn.columns
    print(df_rnn.shape)
    print(columns_rnn)

# if max_train_test_split = True il train jusqu'a derniere moment possible basè sur objective_day
if max_train_test_split:
    if preprocess_xgb:

        # XGB
        fold_train_xgb, fold_test_xgb = train_test_split_XGB_optimized(
            df=df_xgb,
            objective_day=objective_day,
            number_days_to_predict=2,
        )

    if preprocess_rnn:

        # RNN
        fold_train_rnn, fold_test_rnn = train_test_split_RNN_optimized(
            df=df_rnn,
            objective_day=objective_day,
            number_days_to_predict=2,
            input_length=7 * 24,  # 168h lookback
        )
else:
    if preprocess_xgb:

        # XGB
        fold_train_xgb, fold_test_xgb = train_test_split_general(
            df=df_xgb, cutoff=cutoff_day
        )

    if preprocess_rnn:
        # RNN
        fold_train_rnn, fold_test_rnn = train_test_split_general(
            df=df_xgb, cutoff=cutoff_day
        )


        
# # XGB 
# split X and y in both train and test
# standardize X_train and X_test with fit_transform and transform
# create X_train and X_val


# # RNN 
# split X and y in both train and test
# standardize X_train and X_test with fit_transform and transform
# create X_train and X_val

# Remember to denormalize preidciton at the end
