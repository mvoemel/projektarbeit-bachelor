#!/bin/bash

# Activate your environment
conda activate tensorflow-env

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=================="
echo "Starting Job Queue"
echo "=================="

echo "Job 1: Training v1..."
python main.py --version 1 --mode TRAIN --epochs 30 --tag _run1 > logs/v1_train.log 2>&1 || echo "Job 1 Failed"

echo "Job 2: Tuning v2..."
python main.py --version 2 --mode TUNE --trials 10 --epochs 15  > logs/v2_tune.log 2>&1 || echo "Job 2 Failed"

echo "=================="
echo "All Jobs Completed"
echo "=================="