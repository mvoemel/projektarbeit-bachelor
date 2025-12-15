#!/bin/bash

NTFY_TOPIC="your-ntfy-topic-name"

notify() {
    curl -d "$1" ntfy.sh/$NTFY_TOPIC
}

conda activate tensorflow-env
mkdir -p logs

echo "Starting Job Queue..."

python main.py --version 0 --mode TRAIN --optimizer SGD --epochs 30 --tag _baseline --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v0_train.log 2>&1 || notify "v0 TRAIN SGD Failed"
python main.py --version 2 --mode TUNE --optimizer ADAM --epochs 20 --trials 35 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v2_tune_adam.log 2>&1 || notify "v2 TUNE ADAM Failed"

notify "🎉 All Jobs Completed"

echo "All Jobs Completed!"