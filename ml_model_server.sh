#!/bin/bash

python3 src/09_serve_registry_model.py \
  --tracking_uri "http://127.0.0.1:8080" \
  --model_name "model_first" \
  --port 5002 \
  # --version 1
