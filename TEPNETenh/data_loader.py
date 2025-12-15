import tensorflow as tf
import pandas as pd
import numpy as np
import h5py


FEATURE_COLUMNS = [
    'TCR_KF7', 'TCR_KF1', 'TCR_hydrophobicity', 'TCR_aromaticity', 
    'TCR_isoelectric_point', 'TCR_instability_index', 
    'epitope_KF7', 'epitope_KF1','epitope_hydrophobicity', 'epitope_aromaticity',
    'epitope_isoelectric_point', 'epitope_instability_index'
]


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


def create_memory_dataset(X_tcr, X_epi, X_feat, y, batch_size, balanced=False):
    """
    Creates a highly optimized tf.data.Dataset from in-memory arrays.
    Replicates the class balancing logic without touching the disk.
    """
    # 1. Separate Positive and Negative indices
    pos_mask = (y == 1)
    neg_mask = (y == 0)

    # 2. Create datasets for each class
    # We use from_tensor_slices which treats the RAM arrays as the data source
    def make_ds(mask):
        return tf.data.Dataset.from_tensor_slices((
            {
                "TCR_Input": X_tcr[mask],
                "Epitope_Input": X_epi[mask],
                "Physicochemical_Features": X_feat[mask]
            },
            y[mask]
        )).shuffle(buffer_size=len(y[mask])) # Shuffle efficiently in RAM

    pos_ds = make_ds(pos_mask)
    neg_ds = make_ds(neg_mask)

    # 3. Define Sampling Weights (The Balancing Logic)
    if balanced:
        # 50% Positive, 50% Negative
        weights = [0.5, 0.5]
    else:
        # Your logic: approx 1/6 Positive, 5/6 Negative
        weights = [1/6, 5/6]

    # 4. Sample from both datasets to create the final stream
    # This repeats the data indefinitely, so we must define steps_per_epoch later
    dataset = tf.data.Dataset.sample_from_datasets(
        [pos_ds, neg_ds], 
        weights=weights
    )

    # 5. Batch and Prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
