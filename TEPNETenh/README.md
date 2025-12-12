# TEPNETenh

This approach to training a model that predicts the binding of TCR and epitope start with the processed data from the Master project thesis called _TEP-NET: A Deep Neural Network for TCR-Epitope Binding Prediction Using Physicochemical and Embedding Features_ from _Sven De Gasparo_ and _Prof. Dr. Jasmina Bogojeska_ at _Zurich University of Applied Sciences (ZHAW)_ in Switzerland. It assumes that the dataset is 1:5 (pos:neg) imbalanced and uses ProtBERT embeddings of `64` dimensions fitted with `PCA`.

If you want to check out which train/tune runs have been done visit the `log_book.ipynb` notebook. If you want to see the evaluation of the different models visit the `evaluation.ipynb` notebook.

```
TEPNETenh/
├── dataset/
│   ├── processed/
│   │   ├── test_ProtBERT_64_pca.h5
│   │   ├── train_ProtBERT_64_pca.h5
│   │   └── validation_ProtBERT_64_pca.h5
│   └── raw
│       ├── epitope_raw.csv
│       ├── tcr_raw.csv
│       ├── test_raw.csv
│       ├── train_raw.csv
│       └── validation_raw.csv
├── logs/
├── models/                                 # Models (.keras)
├── output/
├── versions/                               # Model definitions
│   ├── __init__.py
│   ├── v0.py
│   └── ...
├── data_loader.py                          # Data generators and loading logic
├── evaluation.ipynb                        # Evaluation notebook of the different models
├── layers.py                               # Custom Layers (PLE, Periodic)
├── log_book.ipynb                          # Log book of the different train / tune runs
├── main.py                                 # The entry point script
├── run_jobs_example.sh                     # Example file to run multiple jobs
└── transformer_block.py                    # Transformer block definition
```

## Get Started

### Prerequisites

The scripts assume you have a `dataset/processed` directory with the following files: `test_ProtBERT_64_pca.h5`, `train_ProtBERT_64_pca.h5`, `validation_ProtBERT_64_pca.h5`.

### Create your Jobs Script

1. **Copy the example file:**

   ```bash
   cp run_jobs_example.sh run_jobs.sh
   ```

2. **Edit your jobs:**
   Open `run_jobs.sh` and modify the job definitions to your liking. Define environment if you are using one (e.g. `conda activate tensorflow-env`)

3. **Make it executable:**

   ```bash
   chmod +x run_jobs.sh
   ```

### Run your Jobs Script

```bash
# Start a new screen session
screen -S training

# Run your script
./run_jobs.sh

# Detach: Press Ctrl+A, then D

# Reattach later
screen -r training

# List sessions
screen -ls

# Terminate the screen session (inside of the screen session)
exit

# Terminate the screen session (outside of the screen session)
screen -X -S training quit
```

To see what is currently happening (live view):

```bash
tail -f logs/v1_train.log
```

_(Press `Ctrl+C` to stop watching; the training continues in the background.)_

## Model Architectures

**Base Model**

- (_v0_) Baseline

**Single Component Additions**

- (_v1_) Baseline + Symmetric Cross-Attention

**Two Component Combinations**

- (_v2_) Baseline + Symmetric Cross-Attention + Transformer Block
- (_v3_) Baseline + Symmetric Cross-Attention + Interaction Map (2D CNN) (**LOW prio**)
- (_v4_) Baseline + Symmetric Cross-Attention + Deep ResNet Classifier Head (**LOW prio**)

**Multi-Component Architectures**

- (_v5_) Baseline + Symmetric Cross-Attention + Interaction Map (2D CNN) + Deep ResNet Classifier Head (**LOW prio**)
- (_v6_) Baseline + Symmetric Cross-Attention + Transformer Block + Interaction Map (2D CNN)
- (_v7_) Baseline + Symmetric Cross-Attention + Transformer Block + Deep ResNet Classifier Head

**Full Architecture**

- (_v8_) Baseline + Symmetric Cross-Attention + Transformer Block + Interaction Map (2D CNN) + Deep ResNet Classifier Head
