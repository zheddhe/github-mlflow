#!/bin/bash
DEFAULT_NAME="random_name_$(date +%Y%m%d%H%M%S)"
RUN_NAME="${3:-$DEFAULT_NAME}"

mlflow run src/ --env-manager=$1 --experiment-id $2 --run-name "$RUN_NAME"