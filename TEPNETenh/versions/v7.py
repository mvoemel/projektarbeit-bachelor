import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, GlobalAveragePooling1D, Dropout, Concatenate, Flatten, Add, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from layers import PiecewiseLinearEncoding, PeriodicEmbeddings
from transformer_block import transformer_block
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
    print("Building Model v7 (Symmetric Cross-Attention + Transformer Block + Deep ResNet Classifier Head)")
    
    bins = np.linspace(0.0, 1.0, num=11)
    PLE = PiecewiseLinearEncoding(bins)
    Periodic = PeriodicEmbeddings(num_frequencies=16)
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    # Path A: TCR context considering Epitope
    tcr_trans_out = transformer_block(
        query=tcr_input, 
        key_value=epitope_input, 
        num_heads=hparams['num_heads'], 
        embed_dim=embed_dim, 
        ff_dim=hparams['ff_dim'], 
        dropout_rate=hparams['dropout_rate']
    )
    pool_tcr = GlobalAveragePooling1D()(tcr_trans_out)

    # Path B: Epitope context considering TCR
    epi_trans_out = transformer_block(
        query=epitope_input, 
        key_value=tcr_input, 
        num_heads=hparams['num_heads'], 
        embed_dim=embed_dim, 
        ff_dim=hparams['ff_dim'], 
        dropout_rate=hparams['dropout_rate']
    )
    pool_epi = GlobalAveragePooling1D()(epi_trans_out)

    x = Concatenate()([pool_tcr, pool_epi])

    if hparams['embed_numerical'] == "PLE":
        feature_embeddings = PLE(feature_input)
    else:
        feature_embeddings = Periodic(feature_input)
        
    feature_embeddings_flatten = Flatten()(feature_embeddings)
    x = Concatenate()([x, feature_embeddings_flatten])

    for _ in range(hparams['num_layers']):
        residual = x
        
        x = Dense(hparams['ff_dim'], activation=hparams['activation'], kernel_regularizer=l2(hparams['l2_reg']))(x)
        x = Dropout(hparams['dropout_rate'])(x)
        x = Dense(hparams['ff_dim'], activation=hparams['activation'], kernel_regularizer=l2(hparams['l2_reg']))(x)
        x = Dropout(hparams['dropout_rate'])(x)
        
        if residual.shape[-1] != x.shape[-1]:
            residual = Dense(x.shape[-1])(residual)
            
        x = Add()([x, residual])
        x = LayerNormalization()(x)

    output = Dense(1, activation="sigmoid", name="Binding_Output")(x)
    
    model = Model(inputs=[tcr_input, epitope_input, feature_input], outputs=output)
    return model
