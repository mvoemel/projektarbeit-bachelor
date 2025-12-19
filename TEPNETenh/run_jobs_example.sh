#!/bin/bash

NTFY_TOPIC="your-ntfy-topic-name"

notify() {
    curl -d "$1" ntfy.sh/$NTFY_TOPIC
}

mkdir -p logs

echo "Starting Job Queue..."

# Baseline
python main.py --version 0 --mode TRAIN --optimizer SGD --epochs 30 --tag _baseline --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v0_train.log 2>&1 || notify "v0 TRAIN SGD Failed"

# Tuning SGD
# python main.py --version 1 --mode TUNE --optimizer SGD --epochs 20 --trials 35 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v1_tune_sgd.log 2>&1 || notify "v1 TUNE SGD Failed"
# python main.py --version 2 --mode TUNE --optimizer SGD --epochs 20 --trials 35 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v2_tune_sgd.log 2>&1 || notify "v2 TUNE SGD Failed"
# python main.py --version 3 --mode TUNE --optimizer SGD --epochs 20 --trials 35 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v3_tune_sgd.log 2>&1 || notify "v3 TUNE SGD Failed"
# python main.py --version 4 --mode TUNE --optimizer SGD --epochs 20 --trials 35 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v4_tune_sgd.log 2>&1 || notify "v4 TUNE SGD Failed"
# python main.py --version 5 --mode TUNE --optimizer SGD --epochs 20 --trials 35 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v5_tune_sgd.log 2>&1 || notify "v5 TUNE SGD Failed"
# python main.py --version 6 --mode TUNE --optimizer SGD --epochs 20 --trials 35 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v6_tune_sgd.log 2>&1 || notify "v6 TUNE SGD Failed"
# python main.py --version 7 --mode TUNE --optimizer SGD --epochs 20 --trials 35 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v7_tune_sgd.log 2>&1 || notify "v7 TUNE SGD Failed"
# python main.py --version 8 --mode TUNE --optimizer SGD --epochs 20 --trials 35 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v8_tune_sgd.log 2>&1 || notify "v8 TUNE SGD Failed"

# Tuning ADAM
# python main.py --version 1 --mode TUNE --optimizer ADAM --epochs 20 --trials 35 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v1_tune_adam.log 2>&1 || notify "v1 TUNE ADAM Failed"
# python main.py --version 2 --mode TUNE --optimizer ADAM --epochs 20 --trials 35 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v2_tune_adam.log 2>&1 || notify "v2 TUNE ADAM Failed"
# python main.py --version 3 --mode TUNE --optimizer ADAM --epochs 20 --trials 35 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v3_tune_adam.log 2>&1 || notify "v3 TUNE ADAM Failed"
# python main.py --version 4 --mode TUNE --optimizer ADAM --epochs 20 --trials 35 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v4_tune_adam.log 2>&1 || notify "v4 TUNE ADAM Failed"
# python main.py --version 5 --mode TUNE --optimizer ADAM --epochs 20 --trials 35 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v5_tune_adam.log 2>&1 || notify "v5 TUNE ADAM Failed"
# python main.py --version 6 --mode TUNE --optimizer ADAM --epochs 20 --trials 35 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v6_tune_adam.log 2>&1 || notify "v6 TUNE ADAM Failed"
# python main.py --version 7 --mode TUNE --optimizer ADAM --epochs 20 --trials 35 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v7_tune_adam.log 2>&1 || notify "v7 TUNE ADAM Failed"
# python main.py --version 8 --mode TUNE --optimizer ADAM --epochs 20 --trials 35 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v8_tune_adam.log 2>&1 || notify "v8 TUNE ADAM Failed"

# Training SGD
# python main.py --version 1 --mode TRAIN --optimizer SGD --epochs 30 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v1_train_sgd.log 2>&1 || notify "v1 TRAIN SGD Failed"
# python main.py --version 2 --mode TRAIN --optimizer SGD --epochs 30 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v2_train_sgd.log 2>&1 || notify "v2 TRAIN SGD Failed"
# python main.py --version 3 --mode TRAIN --optimizer SGD --epochs 30 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v3_train_sgd.log 2>&1 || notify "v3 TRAIN SGD Failed"
# python main.py --version 4 --mode TRAIN --optimizer SGD --epochs 30 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v4_train_sgd.log 2>&1 || notify "v4 TRAIN SGD Failed"
# python main.py --version 5 --mode TRAIN --optimizer SGD --epochs 30 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v5_train_sgd.log 2>&1 || notify "v5 TRAIN SGD Failed"
# python main.py --version 6 --mode TRAIN --optimizer SGD --epochs 30 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v6_train_sgd.log 2>&1 || notify "v6 TRAIN SGD Failed"
# python main.py --version 7 --mode TRAIN --optimizer SGD --epochs 30 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v7_train_sgd.log 2>&1 || notify "v7 TRAIN SGD Failed"
# python main.py --version 8 --mode TRAIN --optimizer SGD --epochs 30 --tag _sgd --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v8_train_sgd.log 2>&1 || notify "v8 TRAIN SGD Failed"

# Training ADAM
# python main.py --version 1 --mode TRAIN --optimizer ADAM --epochs 30 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v1_train_adam.log 2>&1 || notify "v1 TRAIN ADAM Failed"
# python main.py --version 2 --mode TRAIN --optimizer ADAM --epochs 30 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v2_train_adam.log 2>&1 || notify "v2 TRAIN ADAM Failed"
# python main.py --version 3 --mode TRAIN --optimizer ADAM --epochs 30 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v3_train_adam.log 2>&1 || notify "v3 TRAIN ADAM Failed"
# python main.py --version 4 --mode TRAIN --optimizer ADAM --epochs 30 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v4_train_adam.log 2>&1 || notify "v4 TRAIN ADAM Failed"
# python main.py --version 5 --mode TRAIN --optimizer ADAM --epochs 30 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v5_train_adam.log 2>&1 || notify "v5 TRAIN ADAM Failed"
# python main.py --version 6 --mode TRAIN --optimizer ADAM --epochs 30 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v6_train_adam.log 2>&1 || notify "v6 TRAIN ADAM Failed"
# python main.py --version 7 --mode TRAIN --optimizer ADAM --epochs 30 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v7_train_adam.log 2>&1 || notify "v7 TRAIN ADAM Failed"
# python main.py --version 8 --mode TRAIN --optimizer ADAM --epochs 30 --tag _adam --ntfy True --ntfy_topic $NTFY_TOPIC > logs/v8_train_adam.log 2>&1 || notify "v8 TRAIN ADAM Failed"

notify "🎉 All Jobs Completed"

echo "All Jobs Completed!"