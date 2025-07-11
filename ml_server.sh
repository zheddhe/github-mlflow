#!/bin/bash

mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri file:///home/ubuntu/MLflow_Course/mlruns \
  --default-artifact-root file:///home/ubuntu/MLflow_Course/mlruns \
  --serve-artifacts