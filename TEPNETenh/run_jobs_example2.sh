#!/bin/bash

NTFY_TOPIC="your-ntfy-topic-name"

notify() {
    curl -d "$1" ntfy.sh/$NTFY_TOPIC
}

conda activate tensorflow-env
mkdir -p logs

echo "=================="
echo "Starting Job Queue"
echo "=================="

echo "Job 1: Training v0..."
python main.py --version 0 --mode TRAIN --epochs 30 --tag _run1 > logs/v0_train.log 2>&1 && \
    notify "✅ Job 1 Complete: Training v0" || \
    notify "❌ Job 1 Failed: Training v0"

echo "Job 2: Tuning v1..."
python main.py --version 1 --mode TUNE --trials 30 > logs/v1_tune.log 2>&1 && \
    notify "✅ Job 2 Complete: Tuning v1" || \
    notify "❌ Job 2 Failed: Tuning v1"

notify "🎉 All Jobs Completed"

echo "=================="
echo "All Jobs Completed"
echo "=================="