import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import gc
import os
import argparse
import importlib
import optuna
from optuna.samplers import TPESampler
from data_loader import H5DiskGenerator, load_full_data_to_ram


print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")


# Configuration
EMBEDDING_DIM = 64
EMBEDDING_TYPE = 'pca'
IS_BALANCED = False  # Dataset is 1:5 imbalanced
SEED = 42
DATA_PATH = './dataset/processed/'
TRAIN_FILE = f'train_ProtBERT_{EMBEDDING_DIM}_{EMBEDDING_TYPE}.h5'
VALIDATION_FILE = f'validation_ProtBERT_{EMBEDDING_DIM}_{EMBEDDING_TYPE}.h5'
TEST_FILE = f'test_ProtBERT_{EMBEDDING_DIM}_{EMBEDDING_TYPE}.h5'
OUTPUT_PREFIX = './output/'
MODEL_NAME_PREFIX = 'TEPNETenh_model'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True, help='Model version (e.g., 0, 1)')
    parser.add_argument('--mode', type=str, choices=['TRAIN', 'TUNE'], default='TRAIN')
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--tag', type=str, default='', help='Optional tag for filename (e.g. _run1)')
    return parser.parse_args()


def run_tuning(args, model_module):
    print(f"Starting Hyperparameter Tuning for v{args.version}...")
    
    # 1. Load VALIDATION data (Safe to load into RAM)
    val_file = os.path.join(DATA_PATH, VALIDATION_FILE)
    X_val_tcr, X_val_epi, X_val_feat, y_val = load_full_data_to_ram(val_file)

    # 2. Init TRAINING Generator (Streams from Disk)
    train_file = os.path.join(DATA_PATH, TRAIN_FILE)
    train_gen = H5DiskGenerator(train_file, batch_size=32, balanced=IS_BALANCED)
    
    def objective(trial):
        hparams = {
            "batch_size": trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            "learning_rate": trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            "dropout_rate": trial.suggest_float('dropout_rate', 0.0, 0.5),
            "l2_reg": trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
            "ff_dim": trial.suggest_int('ff_dim', 16, 200),
            "num_layers": trial.suggest_int('num_layers', 1, 5),
            "num_heads": trial.suggest_int('num_heads', 2, 50),
            "activation": trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu']),
            "embed_numerical": trial.suggest_categorical('embed_numerical', ['PLE', 'Periodic'])
        }
        
        # Update generator batch size
        train_gen.batch_size = hparams['batch_size']
        
        optimizer = tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'], clipnorm=1.0)

        # Build model dynamically
        model = model_module.create_model(hparams, embed_dim=EMBEDDING_DIM)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='roc_auc')])
        
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        
        history = model.fit(
            train_gen,
            validation_data=({"TCR_Input": X_val_tcr, "Epitope_Input": X_val_epi, "Physicochemical_Features": X_val_feat}, y_val),
            epochs=20, # Reduced for tuning
            verbose=0,
            callbacks=[early_stopping]
        )
        
        val_auc = max(history.history['val_roc_auc'])
        
        # Cleanup
        tf.keras.backend.clear_session()
        del model
        gc.collect()
        
        return val_auc

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.trials)
    print("Best Parameters:", study.best_params)
    
    params_save_name = f"best_params_v{args.version}.txt"
    with open(os.path.join(OUTPUT_PREFIX, params_save_name), "w") as f:
        f.write(str(study.best_params))


def run_training(args, model_module):
    print(f"Starting Training for v{args.version}...")
    
    hparams = model_module.HYPER_PARAMETERS
    print(f"Loaded Hyperparameters: {hparams}")
    
    val_path = os.path.join(DATA_PATH, VALIDATION_FILE)
    X_val_tcr, X_val_epi, X_val_feat, y_val = load_full_data_to_ram(val_path)
    
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    train_gen = H5DiskGenerator(train_path, batch_size=hparams['batch_size'], balanced=False)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'], clipnorm=1.0)
    
    model = model_module.create_model(hparams, embed_dim=EMBEDDING_DIM)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='roc_auc'), Precision(name='precision'), Recall(name='recall')]
    )
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_gen,
        validation_data=({"TCR_Input": X_val_tcr, "Epitope_Input": X_val_epi, "Physicochemical_Features": X_val_feat}, y_val),
        epochs=args.epochs,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # Evaluation
    print("\nLoading Test Set...")
    test_file = os.path.join(DATA_PATH, TEST_FILE)
    X_test_tcr, X_test_epi, X_test_feat, y_test = load_full_data_to_ram(test_file)
    
    print("\nEvaluating on Test Set...")
    results = model.evaluate(
        {"TCR_Input": X_test_tcr, "Epitope_Input": X_test_epi, "Physicochemical_Features": X_test_feat}, 
        y_test, verbose=1, return_dict=True
    )
    
    print(f"\nTest Results ({args.version}):")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    save_name = f"{MODEL_NAME_PREFIX}_v{args.version}_{EMBEDDING_DIM}_{EMBEDDING_TYPE}{args.tag}.keras"
    model.save(os.path.join(OUTPUT_PREFIX, save_name))
    print(f"\nModel saved to {save_name}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_PREFIX, exist_ok=True)
    
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    args = get_args()
    
    try:
        model_module = importlib.import_module(f"versions.v{args.version}")
    except ModuleNotFoundError:
        print(f"Error: Model file 'models/v{args.version}.py' not found.")
        exit(1)
        
    if args.mode == 'TUNE':
        run_tuning(args, model_module)
    else:
        run_training(args, model_module)
