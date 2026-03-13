import glob
import os
import time
import pickle
from power_forecast.params import *


def save_model_ml(model=None, model_name: str = None) -> None:
    """
    Persist trained model locally at f"{LOCAL_REGISTRY_PATH}/{model_name}.pkl".
    If model_name is not provided, uses a timestamp as name.

    Example:
        save_model(model, model_name="histxgb_v1")
        → saved at ~/.lewagon/mlops/models/histxgb_v1.pkl
    """

    if model_name is None:
        model_name = time.strftime("%Y%m%d-%H%M%S")

    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)

    model_path = os.path.join(LOCAL_REGISTRY_PATH, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model saved locally as '{model_name}.pkl'")
    return None


def load_model_ml(model_name: str = None):
    """
    Load a model from LOCAL_REGISTRY_PATH.
    - If model_name is provided (e.g. "histxgb_v1"), loads that specific file.
    - Otherwise, loads the most recent model automatically.

    Example:
        load_model("histxgb_v1")  → charge histxgb_v1.pkl
        load_model()              → charge le plus récent
    """
    if model_name:
        model_path = os.path.join(LOCAL_REGISTRY_PATH, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            print(f"❌ Model '{model_name}.pkl' not found in {LOCAL_REGISTRY_PATH}")
            return None
    else:
        pkl_files = glob.glob(os.path.join(LOCAL_REGISTRY_PATH, "*.pkl"))
        if not pkl_files:
            print("❌ No model found locally")
            return None
        model_path = sorted(pkl_files)[-1]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"✅ Model loaded from {model_path}")
    return model


def save_scaler(scaler, scaler_name: str = 'scaler') -> None:
    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    scaler_path = os.path.join(LOCAL_REGISTRY_PATH, f"{scaler_name}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved locally as '{scaler_name}.pkl'")


def load_scaler(scaler_name: str = 'scaler'):
    scaler_path = os.path.join(LOCAL_REGISTRY_PATH, f"{scaler_name}.pkl")
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler '{scaler_name}.pkl' not found in {LOCAL_REGISTRY_PATH}")
        return None
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"✅ Scaler loaded from {scaler_path}")
    return scaler
