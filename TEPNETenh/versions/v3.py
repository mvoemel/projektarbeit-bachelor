import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, GlobalAveragePooling1D, Dropout, Concatenate, Flatten, Dot, Conv2D, GlobalMaxPooling2D, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from layers import PiecewiseLinearEncoding, PeriodicEmbeddings
import numpy as np

# TODO: hyper param tuning
HYPER_PARAMETERS = {
    "batch_size": 32,
    "learning_rate": 0.0059,
    "dropout_rate": 0.2414,
    "l2_reg": 0.0082,
    "ff_dim": 135,
    "num_layers": 1,
    "num_heads": 50,
    "activation": "tanh",
    "embed_numerical": "PLE"
}

def create_model(hparams, embed_dim=64, feature_dim=12):
    print("Building Model v3 (Symmetric Cross-Attention + Interaction Map (2D CNN))")
    
    bins = np.linspace(0.0, 1.0, num=11)
    PLE = PiecewiseLinearEncoding(bins)
    Periodic = PeriodicEmbeddings(num_frequencies=16)
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    # Branch 1: Symmetric Cross-Attention
    att_tcr_to_epi = MultiHeadAttention(num_heads=hparams['num_heads'], key_dim=embed_dim)(
        query=tcr_input, value=epitope_input
    )
    pool_tcr_to_epi = GlobalAveragePooling1D()(att_tcr_to_epi)
    
    att_epi_to_tcr = MultiHeadAttention(num_heads=hparams['num_heads'], key_dim=embed_dim)(
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
    if hparams['embed_numerical'] == "PLE":
        feature_embeddings = PLE(feature_input)
    else:
        feature_embeddings = Periodic(feature_input)
    feature_embeddings_flatten = Flatten()(feature_embeddings)

    # Combine: Attention contexts + Structural interaction motifs + Physio properties
    x = Concatenate()([pool_tcr_to_epi, pool_epi_to_tcr, pool_cnn, feature_embeddings_flatten])

    for _ in range(hparams['num_layers']):
        x = Dense(hparams['ff_dim'], activation=hparams['activation'], kernel_regularizer=l2(hparams['l2_reg']))(x)
        x = Dropout(hparams['dropout_rate'])(x)

    output = Dense(1, activation="sigmoid", name="Binding_Output")(x)
    
    model = Model(inputs=[tcr_input, epitope_input, feature_input], outputs=output)
    return model
