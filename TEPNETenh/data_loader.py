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
