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

python main.py --version 0 --mode TRAIN --epochs 30 --tag _run1 --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v0_train.log 2>&1 || echo "Job 1 Failed"
python main.py --version 1 --mode TUNE --trials 10 --epochs 15 --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v1_tune.log 2>&1 || echo "Job 2 Failed"

notify "🎉 All Jobs Completed"

echo "=================="
echo "All Jobs Completed"
echo "=================="