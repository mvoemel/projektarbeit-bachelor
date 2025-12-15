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


def create_balanced_memory_generator(X_tcr, X_epi, X_feat, y, batch_size):
    """
    Zero-Copy Generator:
    Instead of copying data into the dataset, we keep the data in global RAM
    and only pass INDICES to the generator.
    """
    # 1. Pre-calculate indices for each class
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    # 2. Define the generator
    def generator():
        while True:
            # A. Select indices for this batch (50/50 balanced)
            # You can tweak this ratio (e.g., n_pos = batch_size // 6)
            n_pos = batch_size // 2
            n_neg = batch_size - n_pos
            
            # Randomly select indices (Fast, lightweight integers)
            batch_pos = np.random.choice(pos_indices, n_pos, replace=False)
            batch_neg = np.random.choice(neg_indices, n_neg, replace=False)
            
            # Combine and shuffle indices
            batch_idx = np.concatenate([batch_pos, batch_neg])
            np.random.shuffle(batch_idx)
            
            # B. Fetch the actual data (Zero-copy slice)
            # This is the only moment data is "touched"
            yield (
                {
                    "TCR_Input": X_tcr[batch_idx],
                    "Epitope_Input": X_epi[batch_idx],
                    "Physicochemical_Features": X_feat[batch_idx]
                },
                y[batch_idx]
            )

    # 3. Create the Dataset from the generator
    output_signature = (
        {
            "TCR_Input": tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
            "Epitope_Input": tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
            "Physicochemical_Features": tf.TensorSpec(shape=(None, 12), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator, 
        output_signature=output_signature
    )
    
    # 4. Prefetch to keep GPU busy
    # This keeps only ~5-10 batches in memory buffer, not the whole dataset
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
