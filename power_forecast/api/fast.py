import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from power_forecast.logic.preprocessing.preprocessor import preproc_histxgb_X_new
from power_forecast.logic.models.registry import load_model_ml

app = FastAPI()
app.state.model = load_model_ml(model_name='HistXGB_v1')

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
def predict(data: dict):
    X_new = pd.DataFrame([data])
    X_preproc = preproc_histxgb_X_new(X_new, column='FRA')
    X_pred = app.state.model.predict(X_preproc)

    return {'prix predit' : float(X_pred)}

@app.get("/")
def root():
    # YOUR CODE HERE
    return {'We are': 'CONNECTED'}
