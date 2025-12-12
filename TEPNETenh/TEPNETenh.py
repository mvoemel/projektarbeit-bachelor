import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, GlobalAveragePooling1D, Dropout, Concatenate, Layer, Flatten, Dot, Conv2D, GlobalMaxPooling2D, Reshape, Add, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import h5py
import gc
import sys
import os
import optuna
from optuna.samplers import TPESampler

print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# ==========================================
#              CONFIGURATION
# ==========================================

RUN_MODE = 'TUNE'  # Options: 'TRAIN', 'TUNE'
MODEL_VERSION = 1 # Options: 0, 1, 2, 3, 4
TAG = '' # e.g. '_id1

# Data & Embedding Settings
EMBEDDING_DIM = 32
EMBEDDING_TYPE = 'pca'
IS_BALANCED = False  # Dataset is 1:5 imbalanced
SEED = 42
EPOCHS = 75
N_TRIALS = 30
DATA_PATH = './dataset/processed/'
TRAIN_FILE = f'train_ProtBERT_{EMBEDDING_DIM}_{EMBEDDING_TYPE}.h5'
VALIDATION_FILE = f'validation_ProtBERT_{EMBEDDING_DIM}_{EMBEDDING_TYPE}.h5'
TEST_FILE = f'test_ProtBERT_{EMBEDDING_DIM}_{EMBEDDING_TYPE}.h5'
OUTPUT_PREFIX = './'
MODEL_NAME = f'TEPNETenh_model_v{MODEL_VERSION}_{EMBEDDING_DIM}_{EMBEDDING_TYPE}{TAG}.keras'

# Training Hyperparameters
HP_BATCH_SIZE = 32
HP_LEARNING_RATE = 0.0059
HP_DROPOUT = 0.2414
HP_L2_REG = 0.0082
HP_FF_DIM = 135
HP_NUM_LAYERS = 1
HP_NUM_HEADS = 50
HP_ACTIVATION = "tanh"
HP_EMBED_NUMERICAL = "PLE"

HYPER_PARAMETERS = {
    "batch_size": 32,
    "learning_rate": 0.0059,
    "dropout": 0.2414,
    "l2_reg": 0.0082,
    "ff_dim": 135,
    "num_layers": 1,
    "num_heads": 50,
    "activation": "tanh",
    "embed_numerical": "PLE"
}

# ==========================================
#           SETUP & UTILS
# ==========================================

np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"Running in {RUN_MODE} mode for version {MODEL_VERSION} with {EMBEDDING_DIM}-dim {EMBEDDING_TYPE} embeddings.")

FEATURE_COLUMNS = [
    'TCR_KF7', 'TCR_KF1', 'TCR_hydrophobicity', 'TCR_aromaticity', 
    'TCR_isoelectric_point', 'TCR_instability_index', 
    'epitope_KF7', 'epitope_KF1','epitope_hydrophobicity', 'epitope_aromaticity',
    'epitope_isoelectric_point', 'epitope_instability_index'
]

def load_h5_metadata_only(h5_path):
    """
    Loads only the lightweight metadata (Labels + Features) into memory.
    Does NOT load the heavy TCR/Epitope embeddings.
    """
    print(f"Loading metadata from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        meta_data = {}
        for key in f['meta'].keys():
            meta_data[key] = f[f'meta/{key}'][:]
    return pd.DataFrame(meta_data)

def load_full_data_to_ram(h5_path):
    """
    Loads EVERYTHING into RAM. Only use this for Validation/Test sets (small).
    """
    print(f"Loading full dataset into RAM: {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        tcr = f['TCR'][:]
        epi = f['epitope'][:]
        
        meta_data = {}
        for key in f['meta'].keys():
            meta_data[key] = f[f'meta/{key}'][:]
            
    df = pd.DataFrame(meta_data)
    
    # Preprocess immediately to save memory copying later
    X_feat = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df['binding'].values.astype(np.float32)
    
    # Convert embeddings to Tensor immediately
    X_tcr = tcr.astype(np.float32)
    X_epi = epi.astype(np.float32)
    
    return X_tcr, X_epi, X_feat, y

# ==========================================
#           DATA GENERATOR (DISK BASED)
# ==========================================

class H5DiskGenerator(tf.keras.utils.Sequence):
    """
    Reads batches from H5 on disk to save RAM.
    Handles class balancing dynamically.
    """
    def __init__(self, h5_path, batch_size, balanced=False):
        super().__init__()  # Fixes the UserWarning about super().__init__
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.balanced = balanced
        
        # 1. Load Metadata into RAM
        self.df_meta = load_h5_metadata_only(h5_path)
        self.y_all = self.df_meta['binding'].values
        self.feat_all = self.df_meta[FEATURE_COLUMNS].values.astype(np.float32)
        
        # 2. Separate Indices
        self.pos_indices = np.where(self.y_all == 1)[0]
        self.neg_indices = np.where(self.y_all == 0)[0]
        
        print(f"Generator initialized: {len(self.pos_indices)} Pos, {len(self.neg_indices)} Neg")
        
        # 3. Open H5 file handle (read-only)
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.tcr_dset = self.h5_file['TCR']
        self.epi_dset = self.h5_file['epitope']
        
        # 4. Define epoch length
        self.steps_per_epoch = int(len(self.pos_indices) / (batch_size // 2 if balanced else (batch_size // 6))) 
        
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        # Determine batch composition
        if self.balanced:
            n_pos = self.batch_size // 2
            n_neg = self.batch_size - n_pos
        else:
            # 1:5 Ratio logic (approx 10 pos, 54 neg for batch 64)
            n_pos = max(1, int(self.batch_size * (1/6)))
            n_neg = self.batch_size - n_pos
            
        # Sample Indices (FIX: replace=False prevents duplicates)
        batch_pos_idx = np.random.choice(self.pos_indices, n_pos, replace=False)
        batch_neg_idx = np.random.choice(self.neg_indices, n_neg, replace=False)
        
        # Combine
        batch_indices = np.concatenate([batch_pos_idx, batch_neg_idx])
        
        # SORT indices for H5 reading (Required by h5py)
        # We sort to fetch from disk, but this ruins the random shuffle order.
        # So we fetch sorted, then we will shuffle the arrays in memory.
        sorted_indices = np.sort(batch_indices)
        
        # Read from Disk
        batch_tcr = self.tcr_dset[sorted_indices].astype(np.float32)
        batch_epi = self.epi_dset[sorted_indices].astype(np.float32)
        
        # Get in-memory data
        batch_feat = self.feat_all[sorted_indices]
        batch_y = self.y_all[sorted_indices].astype(np.float32)
        
        # RESHUFFLE (Crucial step)
        # Since we sorted indices to read from disk, the batch is now ordered by index (likely grouped).
        # We must shuffle the arrays in unison so the model doesn't learn order artifacts.
        shuffle_idxs = np.arange(len(batch_y))
        np.random.shuffle(shuffle_idxs)
        
        return (
            {
                "TCR_Input": batch_tcr[shuffle_idxs], 
                "Epitope_Input": batch_epi[shuffle_idxs], 
                "Physicochemical_Features": batch_feat[shuffle_idxs]
            },
            batch_y[shuffle_idxs]
        )
    
    def on_epoch_end(self):
        pass

# ==========================================
#           CUSTOM LAYERS
# ==========================================

class PiecewiseLinearEncoding(Layer):
    def __init__(self, bins, **kwargs):
        super(PiecewiseLinearEncoding, self).__init__(**kwargs)
        self.bins = tf.convert_to_tensor(bins, dtype=tf.float32)

    def call(self, inputs):
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        bin_widths = self.bins[1:] - self.bins[:-1]
        bin_edges = (inputs_expanded - self.bins[:-1]) / bin_widths
        bin_edges = tf.clip_by_value(bin_edges, 0.0, 1.0)
        return bin_edges

    def get_config(self):
        config = super().get_config()
        config.update({"bins": self.bins.numpy().tolist()})
        return config

class PeriodicEmbeddings(Layer):
    def __init__(self, num_frequencies=16, **kwargs):
        super(PeriodicEmbeddings, self).__init__(**kwargs)
        self.num_frequencies = num_frequencies
        self.freqs = tf.Variable(
            initial_value=tf.random.uniform(shape=(num_frequencies,), minval=0.1, maxval=1.0),
            trainable=True
        )

    def call(self, inputs):
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        periodic_features = tf.concat([
            tf.sin(2 * np.pi * inputs_expanded * self.freqs),
            tf.cos(2 * np.pi * inputs_expanded * self.freqs)
        ], axis=-1)
        return periodic_features

    def get_config(self):
        config = super().get_config()
        config.update({"num_frequencies": self.num_frequencies})
        return config

bins = np.linspace(0.0, 1.0, num=11)
PLE = PiecewiseLinearEncoding(bins)
Periodic = PeriodicEmbeddings(num_frequencies=16)

# ==========================================
#           MODEL ARCHITECTURE
# ==========================================

# Baseline
def create_model_v0(embed_dim, ff_dim, feature_dim, dropout_rate, activation, l2_reg, num_layers, num_heads, embed_numerical):
    print("Using VERSION 0")
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        query=tcr_input, value=epitope_input
    )
    x = GlobalAveragePooling1D()(attention_output)

    if embed_numerical == "PLE":
        feature_embeddings = PLE(feature_input)
    else:
        feature_embeddings = Periodic(feature_input)
        
    feature_embeddings_flatten = Flatten()(feature_embeddings)
    x = Concatenate()([x, feature_embeddings_flatten])

    for _ in range(num_layers):
        x = Dense(ff_dim, activation=activation, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)

    output = Dense(1, activation="sigmoid", name="Binding_Output")(x)
    
    model = Model(inputs=[tcr_input, epitope_input, feature_input], outputs=output)
    return model


# Symmetric Cross-Attention
def create_model_v1(embed_dim, ff_dim, feature_dim, dropout_rate, activation, l2_reg, num_layers, num_heads, embed_numerical):
    print("Using VERSION 1")
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    # 1. TCR attends to Epitope
    att_tcr_to_epi = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        query=tcr_input, value=epitope_input
    )
    pool_tcr = GlobalAveragePooling1D()(att_tcr_to_epi)

    # 2. Epitope attends to TCR
    att_epi_to_tcr = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        query=epitope_input, value=tcr_input
    )
    pool_epi = GlobalAveragePooling1D()(att_epi_to_tcr)

    # 3. Combine
    x = Concatenate()([pool_tcr, pool_epi])

    if embed_numerical == "PLE":
        feature_embeddings = PLE(feature_input)
    else:
        feature_embeddings = Periodic(feature_input)
        
    feature_embeddings_flatten = Flatten()(feature_embeddings)
    x = Concatenate()([x, feature_embeddings_flatten])

    for _ in range(num_layers):
        x = Dense(ff_dim, activation=activation, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)

    output = Dense(1, activation="sigmoid", name="Binding_Output")(x)
    
    model = Model(inputs=[tcr_input, epitope_input, feature_input], outputs=output)
    return model


# Symmetric Cross-Attention + Interaction Map (2D CNN)
def create_model_v2(embed_dim, ff_dim, feature_dim, dropout_rate, activation, l2_reg, num_layers, num_heads, embed_numerical):
    print("Using VERSION 2")
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    # Branch 1: Symmetric Cross-Attention
    att_tcr_to_epi = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        query=tcr_input, value=epitope_input
    )
    pool_tcr_to_epi = GlobalAveragePooling1D()(att_tcr_to_epi)
    
    att_epi_to_tcr = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        query=epitope_input, value=tcr_input
    )
    pool_epi_to_tcr = GlobalAveragePooling1D()(att_epi_to_tcr)

    # Branch 2: Interaction Map (2D CNN)
    # Dot product compares every TCR amino acid with every Epitope amino acid
    # Shape: (Batch, 26, 64)x(Batch, 24, 64) -> (Batch, 26, 24)
    interaction_map = Dot(axes=(2, 2))([tcr_input, epitope_input])
    
    # Add channel dimension for CNN: (Batch, 26, 24, 1)
    interaction_map = Reshape((26, 24, 1))(interaction_map)
    
    # Apply 2D Convolutions to find local binding motifs (e.g. diagonal matches)
    x_cnn = Conv2D(32, (3, 3), activation='relu', padding='valid')(interaction_map)
    x_cnn = Conv2D(64, (3, 3), activation='relu', padding='valid')(x_cnn)
    pool_cnn = GlobalMaxPooling2D()(x_cnn)

    # Branch 3: Physicochemical Features
    if embed_numerical == "PLE":
        feature_embeddings = PLE(feature_input)
    else:
        feature_embeddings = Periodic(feature_input)
    feature_embeddings_flatten = Flatten()(feature_embeddings)

    # Combine: Attention contexts + Structural interaction motifs + Physio properties
    x = Concatenate()([pool_tcr_to_epi, pool_epi_to_tcr, pool_cnn, feature_embeddings_flatten])

    for _ in range(num_layers):
        x = Dense(ff_dim, activation=activation, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)

    output = Dense(1, activation="sigmoid", name="Binding_Output")(x)
    
    model = Model(inputs=[tcr_input, epitope_input, feature_input], outputs=output)
    return model


# Symmetric Cross-Attention + Interaction Map (2D CNN) + Deep ResNet Classifier Head
def create_model_v3(embed_dim, ff_dim, feature_dim, dropout_rate, activation, l2_reg, num_layers, num_heads, embed_numerical):
    print("Using VERSION 3")
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    att_tcr_to_epi = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        query=tcr_input, value=epitope_input
    )
    pool_tcr_to_epi = GlobalAveragePooling1D()(att_tcr_to_epi)

    att_epi_to_tcr = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        query=epitope_input, value=tcr_input
    )
    pool_epi_to_tcr = GlobalAveragePooling1D()(att_epi_to_tcr)

    interaction_map = Dot(axes=(2, 2))([tcr_input, epitope_input])
    interaction_map = Reshape((26, 24, 1))(interaction_map)
    
    x_cnn = Conv2D(32, (3, 3), activation='relu', padding='valid')(interaction_map)
    x_cnn = Conv2D(64, (3, 3), activation='relu', padding='valid')(x_cnn)
    pool_cnn = GlobalMaxPooling2D()(x_cnn)

    if embed_numerical == "PLE":
        feature_embeddings = PLE(feature_input)
    else:
        feature_embeddings = Periodic(feature_input)
    feature_embeddings_flatten = Flatten()(feature_embeddings)

    x = Concatenate()([pool_tcr_to_epi, pool_epi_to_tcr, pool_cnn, feature_embeddings_flatten])

    # Deep Classifier Head (Residual MLP)
    # Using a residual loop allows the network to be deeper without degradation
    for _ in range(num_layers):
        residual = x
        
        # Dense Block
        x = Dense(ff_dim, activation=activation, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(ff_dim, activation=activation, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)
        
        # Match dimensions for residual connection if needed
        if residual.shape[-1] != x.shape[-1]:
            residual = Dense(x.shape[-1])(residual)
            
        x = Add()([x, residual])
        x = LayerNormalization()(x)

    output = Dense(1, activation="sigmoid", name="Binding_Output")(x)

    model = Model(inputs=[tcr_input, epitope_input, feature_input], outputs=output)
    return model


def transformer_block(query, key_value, num_heads, embed_dim, ff_dim, dropout_rate):
    """
    Standard Transformer Block: Attention -> Add & Norm -> FFN -> Add & Norm
    """
    # 1. Multi-Head Attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        query=query, value=key_value
    )
    attn_output = Dropout(dropout_rate)(attn_output)
    
    # 2. Add & Norm (Residual Connection 1)
    # Note: We add query to the output. Shapes must match.
    out1 = LayerNormalization(epsilon=1e-6)(Add()([query, attn_output]))
    
    # 3. Feed Forward Network
    ffn_output = Dense(ff_dim, activation="relu")(out1) # Expand
    ffn_output = Dense(embed_dim)(ffn_output)           # Project back
    ffn_output = Dropout(dropout_rate)(ffn_output)
    
    # 4. Add & Norm (Residual Connection 2)
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ffn_output]))
    
    return out2

# Symmetric Cross-Attention + Transformer Block + Interaction Map (2D CNN) + Deep ResNet Classifier Head
def create_model_v4(embed_dim, ff_dim, feature_dim, dropout_rate, activation, l2_reg, num_layers, num_heads, embed_numerical):
    print("Using VERSION 4")
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    # Branch 1: Symmetric Transformer Blocks
    
    # Path A: TCR context considering Epitope
    tcr_trans_out = transformer_block(
        query=tcr_input, 
        key_value=epitope_input, 
        num_heads=num_heads, 
        embed_dim=embed_dim, 
        ff_dim=ff_dim, 
        dropout_rate=dropout_rate
    )
    pool_tcr = GlobalAveragePooling1D()(tcr_trans_out)

    # Path B: Epitope context considering TCR
    epi_trans_out = transformer_block(
        query=epitope_input, 
        key_value=tcr_input, 
        num_heads=num_heads, 
        embed_dim=embed_dim, 
        ff_dim=ff_dim, 
        dropout_rate=dropout_rate
    )
    pool_epi = GlobalAveragePooling1D()(epi_trans_out)

    # Branch 2: Interaction Map (2D CNN)
    interaction_map = Dot(axes=(2, 2))([tcr_input, epitope_input])
    interaction_map = Reshape((26, 24, 1))(interaction_map)
    
    x_cnn = Conv2D(32, (3, 3), activation='relu', padding='valid')(interaction_map)
    x_cnn = Conv2D(64, (3, 3), activation='relu', padding='valid')(x_cnn)
    pool_cnn = GlobalMaxPooling2D()(x_cnn)

    # Branch 3: Physicochemical Features
    if embed_numerical == "PLE":
        feature_embeddings = PLE(feature_input)
    else:
        feature_embeddings = Periodic(feature_input)
    feature_embeddings_flatten = Flatten()(feature_embeddings)

    x = Concatenate()([pool_tcr, pool_epi, pool_cnn, feature_embeddings_flatten])

    for _ in range(num_layers):
        residual = x
        
        x = Dense(ff_dim, activation=activation, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(ff_dim, activation=activation, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)
        
        if residual.shape[-1] != x.shape[-1]:
            residual = Dense(x.shape[-1])(residual)
            
        x = Add()([x, residual])
        x = LayerNormalization()(x)

    output = Dense(1, activation="sigmoid", name="Binding_Output")(x)

    model = Model(inputs=[tcr_input, epitope_input, feature_input], outputs=output)
    return model

# ==========================================
#           MAIN EXECUTION
# ==========================================

# 1. Load VALIDATION data (Safe to load into RAM)
val_file = os.path.join(DATA_PATH, VALIDATION_FILE)
X_val_tcr, X_val_epi, X_val_feat, y_val = load_full_data_to_ram(val_file)

# 2. Init TRAINING Generator (Streams from Disk)
train_file = os.path.join(DATA_PATH, TRAIN_FILE)
train_gen = H5DiskGenerator(train_file, HP_BATCH_SIZE, balanced=IS_BALANCED)

MODEL_CREATORS = {
    0: create_model_v0,
    1: create_model_v1,
    2: create_model_v2,
    3: create_model_v3,
    4: create_model_v4,
}
create_model = MODEL_CREATORS.get(MODEL_VERSION)
if create_model is None:
    raise ValueError(f"Unknown MODEL_VERSION: {MODEL_VERSION}")

if RUN_MODE == 'TUNE':
    print("Starting Hyperparameter Tuning...")
    def objective(trial):
        ff_dim = trial.suggest_int('ff_dim', 16, 200)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu'])
        num_layers = trial.suggest_int('num_layers', 1, 10)
        num_heads = trial.suggest_int('num_heads', 2, 50)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        embed_numerical = trial.suggest_categorical('embed_numerical', ['PLE', 'Periodic'])
        
        # Update generator batch size
        train_gen.batch_size = batch_size
        
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, clipnorm=1.0)
        model = create_model(EMBEDDING_DIM, ff_dim, 12, dropout_rate, activation, l2_reg, num_layers, num_heads, embed_numerical)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='roc_auc')])
        
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        
        history = model.fit(
            train_gen,
            validation_data=({"TCR_Input": X_val_tcr, "Epitope_Input": X_val_epi, "Physicochemical_Features": X_val_feat}, y_val),
            epochs=20,
            verbose=0,
            callbacks=[early_stopping]
        )
        
        val_auc = max(history.history['val_roc_auc'])
        tf.keras.backend.clear_session()
        gc.collect()
        return val_auc

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    print("Best Parameters:", study.best_params)

elif RUN_MODE == 'TRAIN':
    print("Starting Training...")
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=HP_LEARNING_RATE, clipnorm=1.0)
    
    model = create_model(
        EMBEDDING_DIM, HP_FF_DIM, 12, HP_DROPOUT, HP_ACTIVATION, 
        HP_L2_REG, HP_NUM_LAYERS, HP_NUM_HEADS, HP_EMBED_NUMERICAL
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='roc_auc'), Precision(name='precision'), Recall(name='recall')]
    )
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_gen,
        validation_data=({"TCR_Input": X_val_tcr, "Epitope_Input": X_val_epi, "Physicochemical_Features": X_val_feat}, y_val),
        epochs=EPOCHS,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # --- Evaluation ---
    # Load Test Data into RAM (Test is usually small enough, ~1.5GB)
    print("\nLoading Test Set...")
    test_file = os.path.join(DATA_PATH, TEST_FILE)
    X_test_tcr, X_test_epi, X_test_feat, y_test = load_full_data_to_ram(test_file)
    
    print("\nEvaluating on Test Set...")
    results = model.evaluate(
        {"TCR_Input": X_test_tcr, "Epitope_Input": X_test_epi, "Physicochemical_Features": X_test_feat}, 
        y_test, verbose=1, return_dict=True
    )
    
    print(f"\nTest Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
        
    model_name = os.path.join(OUTPUT_PREFIX, MODEL_NAME)
    model.save(model_name)
    print(f"\nModel saved to {model_name}")
