#!/bin/bash

# Activate your environment
conda activate tensorflow-env

# Create logs directory if it doesn't exist
mkdir -p logs

echo "======================"
echo "Starting Job Queue"
echo "======================"

# Job 1: Train Version 1
echo "Job 1: Training v1..."
python main.py --version 1 --mode TRAIN --epochs 75 --tag _run1 > logs/v1_train.log 2>&1 || echo "Job 1 Failed"

# Job 2: Tune Version 2
echo "Job 2: Tuning v2..."
python main.py --version 2 --mode TUNE --trials 30 > logs/v2_tune.log 2>&1 || echo "Job 2 Failed"

echo "======================"
echo "All Jobs Completed"
echo "======================"
