# Imports librairies
from mlflow import MlflowClient
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import argparse
import os

def main():
    # Get project root directory (one level up from script location)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                       default=os.path.join(PROJECT_ROOT, "data", "fake_data.csv"),
                       help='path to the data file')
    args = parser.parse_args()

    # Define tracking_uri (localhost)
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

    # Define experiment name, run name and artifact_path name
    apple_experiment = mlflow.set_experiment("Apple_Models")
    run_name = "third_run_repro_first_run"
    artifact_path = "rf_apples"

    # Import Database
    data = pd.read_csv(args.data_path)
    X = data.drop(columns=["date", "demand"])
    X = X.astype('float')
    y = data["demand"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    params = {
        "n_estimators": 10,
        "max_depth": 10,
        "random_state": 42,
    }

    rf = RandomForestRegressor(**params)  # type: ignore
    rf.fit(X_train, y_train)

    # Evaluate model
    y_pred = rf.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    # Store information in tracking server
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=rf, input_example=X_val, artifact_path=artifact_path
        )

if __name__ == "__main__":
    main()