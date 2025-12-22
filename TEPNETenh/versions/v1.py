import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, GlobalAveragePooling1D, Dropout, Concatenate, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from layers import PiecewiseLinearEncoding, PeriodicEmbeddings
import numpy as np

# using SGD optimizer (better AUC)
HYPER_PARAMETERS = {
    'batch_size': 16, 
    'learning_rate': 0.005674018095876286, 
    'dropout_rate': 0.357649350688716, 
    'l2_reg': 0.00018082955891304315, 
    'ff_dim': 87, 
    'num_layers': 2, 
    'num_heads': 38, 
    'activation': 'tanh', 
    'embed_numerical': 'PLE'
}

# using ADAM optimizer
# HYPER_PARAMETERS = {
#     'batch_size': 64, 
#     'learning_rate': 0.008781408196485976, 
#     'dropout_rate': 0.4812236474710556, 
#     'l2_reg': 5.69307476764461e-05, 
#     'ff_dim': 107, 
#     'num_layers': 2, 
#     'num_heads': 15, 
#     'activation': 'tanh', 
#     'embed_numerical': 'Periodic'
# }

def create_model(hparams, embed_dim=64, feature_dim=12):
    print("Building Model v1 (Symmetric Cross-Attention)")
    
    bins = np.linspace(0.0, 1.0, num=11)
    PLE = PiecewiseLinearEncoding(bins)
    Periodic = PeriodicEmbeddings(num_frequencies=16)
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    # 1. TCR attends to Epitope
    att_tcr_to_epi = MultiHeadAttention(num_heads=hparams['num_heads'], key_dim=embed_dim)(
        query=tcr_input, value=epitope_input
    )
    pool_tcr = GlobalAveragePooling1D()(att_tcr_to_epi)

    # 2. Epitope attends to TCR
    att_epi_to_tcr = MultiHeadAttention(num_heads=hparams['num_heads'], key_dim=embed_dim)(
        query=epitope_input, value=tcr_input
    )
    pool_epi = GlobalAveragePooling1D()(att_epi_to_tcr)

    # 3. Combine
    x = Concatenate()([pool_tcr, pool_epi])

    if hparams['embed_numerical'] == "PLE":
        feature_embeddings = PLE(feature_input)
    else:
        feature_embeddings = Periodic(feature_input)
        
    feature_embeddings_flatten = Flatten()(feature_embeddings)
    x = Concatenate()([x, feature_embeddings_flatten])

    for _ in range(hparams['num_layers']):
        x = Dense(hparams['ff_dim'], activation=hparams['activation'], kernel_regularizer=l2(hparams['l2_reg']))(x)
        x = Dropout(hparams['dropout_rate'])(x)

    output = Dense(1, activation="sigmoid", name="Binding_Output")(x)
    
    model = Model(inputs=[tcr_input, epitope_input, feature_input], outputs=output)
    return model
