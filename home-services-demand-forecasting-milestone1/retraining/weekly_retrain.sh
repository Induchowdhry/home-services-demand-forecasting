#!/bin/bash
set -e
echo "Starting weekly demand-model retraining..."
python ../src/train.py
echo "Retraining completed."
