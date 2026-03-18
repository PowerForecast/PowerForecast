from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    Dense,
    SimpleRNN,
    Normalization,
    LSTM,
    Dropout,
    LayerNormalization,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from datetime import datetime
import random
from power_forecast.logic.get_data.build_dataframe import build_common_dataframe, add_features_XGB, add_features_RNN
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from power_forecast.params import *
from power_forecast.logic.get_data.build_dataframe import (
    build_common_dataframe,
    add_features_RNN,
)
from power_forecast.logic.preprocessing.train_test_split import (
    train_test_split_general,
    train_test_split_RNN_optimized,
)
from power_forecast.logic.preprocessing.split_X_y_standardize import (
    get_X_y_vectorized_RNN,
    get_Xi_yi_single_sequence,
)
from power_forecast.logic.utils.graphs import plot_predictions_rnn, plot_best_predictions
#pd.set_option("display.max_columns", None)

print('       ---------MODEL 3 Alessandro-------------')
## PARAMÈTRES DE STRATÉGIE D'ENTRAÎNEMENT
max_train_test_split = True

objective_day = pd.Timestamp("2024-03-16", tz="UTC")


# Other inputs
input_length = 21 * 24  # 3 weeks context fed to RNN
stride_sequences = 24 * 3  # doit etre plus haute que output length
prediction_horizon_days = 2
country_price_objective = "France"
prediction_length = prediction_horizon_days * 24  # predict 48h of target day

#Model hyperparam
train_new_model = True
patience_model = 20
batch_size_model = 16
epochs_model = 100
version = "3"





def initialize_model_lstm(input_shape, output_length):
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(
                128, activation="tanh", return_sequences=True, recurrent_dropout=0.1
            ),  # dropout on recurrent connections
            Dropout(0.2),
            LSTM(64, activation="tanh", return_sequences=True, recurrent_dropout=0.1),
            LSTM(32, activation="tanh", return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            LayerNormalization(),
            Dense(32, activation="relu"),
            Dense(output_length, activation="linear"),
        ]
    )
    optimizer = Adam(learning_rate=1e-3, clipvalue=0.5)
    model.compile(
        optimizer=optimizer,
        loss=Huber(delta=1.0),  # behaves like MAE for large errors, MSE for small ones
        metrics=["mae", "mse"],
    )
    model.summary()
    return model


df_common = build_common_dataframe(
    filepath="raw_data/all_countries.csv",
    country_objective=country_price_objective,
    target_day_distance=prediction_horizon_days,
    time_interval="h",
    keep_only_neighbors=True,
    add_meteo=True,
    add_crisis=True,
    add_entsoe=True,
)

df = add_features_RNN(
    df=df_common,
    country_objective=country_price_objective,
    target_day_distance=prediction_horizon_days,
    add_future_time_features=True,
    add_future_meteo=True,
)

columns_rnn = df.columns
print(f"      Shape of all data{df.shape}")



fold_train_rnn, fold_test_rnn = train_test_split_RNN_optimized(
    df=df,
    objective_day=objective_day,
    number_days_to_predict=prediction_horizon_days,
    input_length=input_length,  # 168h lookback
)

scaler = StandardScaler()

# ── Train ──────────────────────────────────────────────────────────────────
X_train, y_train = get_X_y_vectorized_RNN(
    fold=fold_train_rnn,
    feature_cols=fold_train_rnn.columns,
    country_objective=country_price_objective,
    stride=stride_sequences,
    input_length=input_length,
    output_length=prediction_length,
    scaler=scaler,
    fit_scaler=True,
)

# ── Test ───────────────────────────────────────────────────────────────────
X_new, y_true = get_X_y_vectorized_RNN(
    fold=fold_test_rnn,
    feature_cols=fold_test_rnn.columns,
    country_objective=country_price_objective,
    stride=stride_sequences,
    input_length=input_length,
    output_length=prediction_length,
    scaler=scaler,
    fit_scaler=False,
)

# SAve X_new and y_true
RNN_X_NEW_DATA_DIR = Path("power_forecast/donnees/x_new_rnn")
RNN_X_NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
RNN_Y_TRUE_DATA_DIR = Path("power_forecast/donnees/y_true_rnn")
RNN_Y_TRUE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Compute datetime indices ---
X_new_index = pd.date_range(
    start=objective_day - pd.Timedelta(hours=input_length),
    end=objective_day - pd.Timedelta(hours=1),
    freq="H"
)

y_true_index = pd.date_range(
    start=objective_day,
    periods=prediction_length,
    freq="H"
)

# --- Print indices ---
print("X_new datetime index:")
print(X_new_index)

print("\ny_true datetime index:")
print(y_true_index)

# --- Compute string representations for filenames ---
date_start_X_str = X_new_index[0].strftime("%Y-%m-%d")
date_end_X_str   = X_new_index[-1].strftime("%Y-%m-%d")

date_start_y_str = y_true_index[0].strftime("%Y-%m-%d")
date_end_y_str   = y_true_index[-1].strftime("%Y-%m-%d")

# --- Build file paths ---
x_new_path = RNN_X_NEW_DATA_DIR / f"X_new_{date_start_X_str}_{date_end_X_str}_{X_new.shape[2]}f_rnn.npy"
y_true_path = RNN_Y_TRUE_DATA_DIR / f"y_true_{date_start_y_str}_{date_end_y_str}_{prediction_length}h_rnn.npy"

# ✅ Add this after building the paths:
np.save(x_new_path, X_new)
np.save(y_true_path, y_true)
print(f"✅ X_new saved → {x_new_path}")
print(f"✅ y_true saved → {y_true_path}")


# # ── X_new : dernière séquence du fold_test pour prédiction ────────────────
# X_new = X_test[-1:]  # (1, input_length, n_features) -> deja bon dimension

print("📐 Shapes finales :")
print(f"    X_train: {X_train.shape} → (n_seq, input_length, n_features)")
print(f"    y_train: {y_train.shape} → (n_seq, output_length)")
print(f"    X_new: {X_new.shape} → (1, input_length, n_features)")
print(f"    y_true: {y_true.shape}→ (n_seq, output_length)")


input_shape=X_train.shape[1:]
output_length=y_train.shape[1]



# Model RNN 

MODELS_DIR = Path("power_forecast/donnees/saved_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RNN_MODEL_NAME    = "model_lstm_3weeks_57f_wed.keras"
model_path_rnn    = MODELS_DIR / RNN_MODEL_NAME
RNN_N_FEATURES    = 57



early_stopping = EarlyStopping(
    monitor="loss", patience=patience_model, restore_best_weights=True
)


if model_path_rnn.exists():
    print(f"✅ Modèle existant chargé : {model_path_rnn}")
    model_lstm = load_model(model_path_rnn)

else:
    print(f"🏋️ Aucun modèle trouvé — entraînement en cours...")
    model_lstm = initialize_model_lstm(input_shape=input_shape, output_length=output_length)

    history = model_lstm.fit(
        X_train,
        y_train,
        epochs=epochs_model,
        batch_size=batch_size_model,
        callbacks=[early_stopping],
        verbose=1,
    )
    model_lstm.save(model_path_rnn)
    print(f"✅ Modèle sauvegardé : {model_path_rnn}")


if max_train_test_split:
    y_pred_rnn  = model_lstm.predict(X_new, verbose=0).flatten()
    y_true_flat = y_true.flatten()

    time_index  = pd.date_range(
        start   = objective_day,
        periods = output_length,
        freq    = "h",
        tz      = "UTC"
    )
    y_true_plot = y_true_flat
    y_pred_plot = y_pred_rnn


# ── Métriques (toutes séquences) ──────────────────────────────────────────
mae  = mean_absolute_error(y_true_flat, y_pred_rnn)
mse  = mean_squared_error(y_true_flat, y_pred_rnn)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true_flat - y_pred_rnn) / (y_true_flat + 1e-8))) * 100

print("\n📐 Métriques finales :")
print(f"   MAE  : {mae:.4f}")
print(f"   MSE  : {mse:.4f}")
print(f"   RMSE : {rmse:.4f}")
print(f"   MAPE : {mape:.2f} %")

