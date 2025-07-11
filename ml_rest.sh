#!/bin/bash

mlflow models serve \
        --model-uri '/home/ubuntu/MLflow_Course/mlruns/637792679469892621/9f1a36320fb34910841e16fa7cef294d/artifacts/rf_apples' \
        --port 5002 \
        --host 0.0.0.0 \
        --env-manager local