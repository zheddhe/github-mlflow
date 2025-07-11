import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
from scipy.stats import randint

def load_and_prep_data(data_path: str):
    """Load and prepare data for training."""
    data = pd.read_csv(data_path)
    X = data.drop(columns=["date", "demand"])
    X = X.astype('float')
    y = data["demand"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Basic setup
    EXPERIMENT_NAME = "RandomizedSearchCV_Random_Forest"
    N_TRIALS = 5

    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Handle experiment creation/deletion
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment and experiment.lifecycle_stage == 'deleted':
        # If experiment exists but is deleted, create a new one with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        EXPERIMENT_NAME = f"{EXPERIMENT_NAME}_{timestamp}"
        client.create_experiment(EXPERIMENT_NAME)
    elif experiment is None:
        # If experiment doesn't exist, create it
        client.create_experiment(EXPERIMENT_NAME)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Enable autologging
    mlflow.sklearn.autolog(
        log_models=True
    )

    # Load data
    X_train, X_val, y_train, y_val = load_and_prep_data("data/fake_data.csv")

    # Define parameter search space
    param_distributions = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4),
    }

    # Create and run RandomizedSearchCV
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=N_TRIALS,
        cv=5,
        scoring='r2',
        random_state=42
    )

    # Fit the model - autolog will automatically create the runs
    search.fit(X_train, y_train)

    # Get best run info
    best_params = search.best_params_
    best_score = search.best_score_

    # Find the best run from MLflow
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],  # type: ignore
        order_by=["metrics.rmse ASC"],
        max_results=1,  # best run only
        filter_string="",
    )
    # find the params and name of the best run
    best_run = runs[0] if runs else None
    if best_run:
        best_params = best_run.data.params
    best_run_name = best_run.data.tags.get('mlflow.runName', 'Not found') if best_run else 'Not found'  # type: ignore

    # find the parent run information
    parent_id = best_run.data.tags.get("mlflow.parentRunId")  # type: ignore
    parent_run = client.get_run(parent_id) if parent_id else None

    # Create a summary of results with better formatting
    summary = f"""Random Forest Trials Summary:
---------------------------
🏆 Best Experiment Name: {EXPERIMENT_NAME}
🎯 Best Run Name: {best_run_name}

Best Model Parameters:
🌲 Number of Trees: {best_params['n_estimators']}
📏 Max Tree Depth: {best_params['max_depth']}
📎 Min Samples Split: {best_params['min_samples_split']}
🍂 Min Samples Leaf: {best_params['min_samples_leaf']}
📊 Best CV Score: {best_score:.4f}
"""

    # Log summary to the parent run
    with mlflow.start_run(run_id=parent_run.info.run_id):  # type: ignore

        # Log summary as an artifact
        with open("summary_proposal.txt", "w") as f:
            f.write(summary)
        mlflow.log_artifact("summary_proposal.txt")

if __name__ == "__main__":
    main()
