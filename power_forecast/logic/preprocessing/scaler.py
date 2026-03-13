import pandas as pd
from sklearn.preprocessing import StandardScaler

def standard_scaling(df):
    scaler = StandardScaler().set_output(transform='pandas')
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler

def standard_scaling_X_new(df, scaler):
    df_scaled = scaler.transform(df)
    return df_scaled
