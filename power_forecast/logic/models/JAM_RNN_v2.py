import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from power_forecast.logic.get_data.build_dataframe import build_feature_dataframe
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers, Input, optimizers
from tensorflow.keras.layers import Normalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

pd.set_option('display.max_columns', None)

# ================================================================= #
# 1. CHARGEMENT DES DONNÉES                                         #
# ================================================================= #

df = build_feature_dataframe(
    filepath='raw_data/all_countries.csv',
    load_from_pickle = False,
    country_objective='France',
    target_day_distance=2,
    time_interval='h',
    save_name='df_with_features',
    drop_nan=True,
    keep_only_neighbors=True,
    add_lag_frontiere=True,
    add_crisis=True,
    add_gen_load_forecast=True,
    add_catch24=True,
)

# ================================================================= #
# 2. FEATURE SELECTION PAR LASSO                                    #
# ================================================================= #

def lasso_feature_selection(df: pd.DataFrame, target: str,
                             alpha: float = 0.01) -> List[str]:
    features = [c for c in df.columns if c != target]
    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_scaled, y)

    selected = [features[i] for i, coef in enumerate(lasso.coef_) if coef != 0]
    print(f"\nLasso feature selection (alpha={alpha}) :")
    print(f"  {len(features)} features initiales → {len(selected)} features sélectionnées")
    print(f"  Features conservées : {selected}")
    return selected


selected_features = lasso_feature_selection(df, target='FRA', alpha=0.01)

df_selected = df[selected_features + ['FRA']]
print(f"\nShape après feature selection : {df_selected.shape}")

# ================================================================= #
# 3. CONFIGURATION GLOBALE                                          #
# ================================================================= #

TARGET          = 'FRA'
N_FEATURES      = df_selected.shape[1]

FOLD_LENGTH      = 24 * 365 * 5
FOLD_STRIDE      = 24 * 7
TRAIN_TEST_RATIO = 1 - (48 / FOLD_LENGTH)

INPUT_LENGTH    = 24 * 14
OUTPUT_LENGTH   = 1
SEQUENCE_STRIDE = 24
DAY_AHEAD_GAP   = 24

print(f"N_FEATURES après sélection : {N_FEATURES}")
print(f"INPUT_LENGTH = {INPUT_LENGTH} pas = {INPUT_LENGTH//24} jours")

# ================================================================= #
# 4. FONCTIONS DE DÉCOUPAGE                                         #
# ================================================================= #

def get_folds(df: pd.DataFrame, fold_length: int, fold_stride: int) -> List[pd.DataFrame]:
    folds = []
    for idx in range(0, len(df), fold_stride):
        if (idx + fold_length) > len(df):
            break
        folds.append(df.iloc[idx:idx + fold_length, :])
    return folds


def train_test_split(fold: pd.DataFrame, train_test_ratio: float,
                     input_length: int) -> Tuple[pd.DataFrame]:
    last_train_idx = round(train_test_ratio * len(fold))
    fold_train     = fold.iloc[0:last_train_idx, :]
    fold_test      = fold.iloc[last_train_idx - input_length:, :]
    return (fold_train, fold_test)


def get_X_y_strides(fold: pd.DataFrame, input_length: int, output_length: int,
                    sequence_stride: int) -> Tuple[np.array]:
    X, y = [], []
    for i in range(0, len(fold), sequence_stride):
        if (i + input_length + DAY_AHEAD_GAP + output_length) > len(fold):
            break
        X.append(fold.iloc[i:i + input_length, :])
        y_start = i + input_length + DAY_AHEAD_GAP
        y.append(fold.iloc[y_start:y_start + output_length, :][[TARGET]])
    return (np.array(X), np.array(y))


# ================================================================= #
# 5. ARCHITECTURE LSTM-LSTM ENCODER-DECODER                         #
# ================================================================= #

def init_model(X_train: np.array, y_train: np.array) -> tf.keras.Model:
    normalizer = Normalization()
    normalizer.adapt(X_train)

    output_length = y_train.shape[1]

    model = models.Sequential([
        Input(shape=X_train[0].shape),
        normalizer,
        layers.LSTM(300, activation='tanh', return_sequences=False),
        layers.Dropout(0.2),
        layers.RepeatVector(1),
        layers.LSTM(100, activation='tanh', return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(output_length, activation='linear'),
    ])

    model.compile(
        loss='mse',
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=['mae']
    )
    return model


# ================================================================= #
# 6. ENTRAÎNEMENT                                                   #
# ================================================================= #

def fit_model(model: tf.keras.Model, X_train: np.array, y_train: np.array,
              verbose: int = 1) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, mode='min',
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-6, verbose=verbose),
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=0.3,
        shuffle=False,
        batch_size=16,
        epochs=100,
        callbacks=callbacks,
        verbose=verbose
    )
    return model, history


# ================================================================= #
# 7. MÉTRIQUES                                                      #
# ================================================================= #

def compute_metrics(y_true: np.array, y_pred: np.array) -> Dict[str, float]:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mae   = np.mean(np.abs(y_true - y_pred))
    rmse  = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape  = 100 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)))
    smape = 100 * np.mean(np.abs(y_true - y_pred) /
                          ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8))
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
            'MAPE': round(mape, 4), 'SMAPE': round(smape, 4)}


def print_metrics(metrics: Dict[str, float], label: str = ""):
    print(f"  {label:10s} | MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} "
          f"| MAPE={metrics['MAPE']:.2f}% | SMAPE={metrics['SMAPE']:.2f}%")


# ================================================================= #
# 8. BASELINE NAÏF                                                  #
# ================================================================= #

def init_baseline() -> tf.keras.Model:
    target_idx = list(df_selected.columns).index(TARGET)
    model = models.Sequential([
        layers.Lambda(lambda x: x[:, -1, target_idx, None])
    ])
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.02), metrics=['mae'])
    return model


# ================================================================= #
# 9. VISUALISATION                                                  #
# ================================================================= #

def plot_history(history: tf.keras.callbacks.History):
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    for a, key, title in zip(ax,
                              ['loss', 'mae', 'lr'],
                              ['Perte MSE', 'Métrique MAE', 'Learning Rate']):
        if key in history.history:
            a.plot(history.history[key], label='Train')
            if f'val_{key}' in history.history:
                a.plot(history.history[f'val_{key}'], label='Validation')
            a.set_title(title)
            a.set_xlabel('Époque')
            if key == 'lr':
                a.set_yscale('log')
            a.legend()
            a.grid(linewidth=0.5)
    plt.tight_layout()
    return ax


# ================================================================= #
# 10. PIPELINE PRINCIPAL — UN SEUL FOLD                             #
# ================================================================= #

folds = get_folds(df_selected, FOLD_LENGTH, FOLD_STRIDE)
print(f'\n{len(folds)} folds générés, forme : {folds[0].shape}')

fold = folds[0]
(fold_train, fold_test) = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)

X_train, y_train = get_X_y_strides(fold_train, INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)
X_test,  y_test  = get_X_y_strides(fold_test,  INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

print(f"X_train : {X_train.shape} | y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}  | y_test  : {y_test.shape}")

# Baseline
baseline_model   = init_baseline()
y_pred_baseline  = baseline_model.predict(X_test, verbose=0)
metrics_baseline = compute_metrics(y_test, y_pred_baseline)

# LSTM Encoder-Decoder
model, history = fit_model(init_model(X_train, y_train), X_train, y_train, verbose=1)
y_pred   = model.predict(X_test, verbose=0)
metrics  = compute_metrics(y_test, y_pred)

print(f"\n{'='*65}")
print_metrics(metrics_baseline, label="Baseline")
print_metrics(metrics,          label="LSTM")
print(f"  🔥 Amélioration MAE : "
      f"{round((1 - metrics['MAE'] / metrics_baseline['MAE']) * 100, 2)} %")
print(f"{'='*65}")

plot_history(history)


# ================================================================= #
# 11. VALIDATION CROISÉE                                            #
# ================================================================= #

def cross_validate() -> pd.DataFrame:
    results = []
    folds   = get_folds(df_selected, FOLD_LENGTH, FOLD_STRIDE)

    for fold_id, fold in enumerate(folds):
        print(f"\n{'═'*65}")
        print(f"FOLD {fold_id + 1}/{len(folds)}")

        (fold_train, fold_test) = train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)
        X_tr, y_tr = get_X_y_strides(fold_train, INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)
        X_te, y_te = get_X_y_strides(fold_test,  INPUT_LENGTH, OUTPUT_LENGTH, SEQUENCE_STRIDE)

        # Baseline
        bm         = init_baseline()
        y_pred_bl  = bm.predict(X_te, verbose=0)
        metrics_bl = compute_metrics(y_te, y_pred_bl)
        print_metrics(metrics_bl, label="Baseline")

        # LSTM
        m, _    = fit_model(init_model(X_tr, y_tr), X_tr, y_tr, verbose=0)
        y_pred  = m.predict(X_te, verbose=0)
        metrics = compute_metrics(y_te, y_pred)
        print_metrics(metrics, label="LSTM")
        print(f"  🏋🏽 Amélioration MAE : "
              f"{round((1 - metrics['MAE'] / metrics_bl['MAE']) * 100, 2)} %")

        results.append({
            'fold':          fold_id + 1,
            'baseline_mae':  metrics_bl['MAE'],  'baseline_smape': metrics_bl['SMAPE'],
            'lstm_mae':      metrics['MAE'],      'lstm_rmse':      metrics['RMSE'],
            'lstm_mape':     metrics['MAPE'],     'lstm_smape':     metrics['SMAPE'],
            'improvement_%': round((1 - metrics['MAE'] / metrics_bl['MAE']) * 100, 2)
        })

    df_results = pd.DataFrame(results)

    print(f"\n{'═'*65}")
    print("RÉCAPITULATIF VALIDATION CROISÉE")
    print(df_results.to_string(index=False))
    print(f"\nMoyenne LSTM SMAPE   : {df_results['lstm_smape'].mean():.2f}%")
    print(f"Amélioration moyenne : {df_results['improvement_%'].mean():.2f}%")

    return df_results


df_results = cross_validate()
