import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, GlobalAveragePooling1D, Dropout, Concatenate, Flatten, Add, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from layers import PiecewiseLinearEncoding, PeriodicEmbeddings
import numpy as np

# using SGD optimizer (better AUC)
HYPER_PARAMETERS = {
    'batch_size': 16, 
    'learning_rate': 0.006393277610030755, 
    'dropout_rate': 0.3565391348813297, 
    'l2_reg': 0.0003510299681926548, 
    'ff_dim': 156, 
    'num_layers': 2, 
    'num_heads': 30, 
    'activation': 'leaky_relu', 
    'embed_numerical': 'PLE'
}

# using ADAM optimizer
# HYPER_PARAMETERS = {
#     'batch_size': 128, 
#     'learning_rate': 0.00853618986286683, 
#     'dropout_rate': 0.40419867405823057, 
#     'l2_reg': 8.200518402245828e-05, 
#     'ff_dim': 34, 
#     'num_layers': 4, 
#     'num_heads': 23, 
#     'activation': 'tanh', 
#     'embed_numerical': 'PLE'
# }

def create_model(hparams, embed_dim=64, feature_dim=12):
    print("Building Model v4 (Symmetric Cross-Attention + Deep ResNet Classifier Head)")
    
    bins = np.linspace(0.0, 1.0, num=11)
    PLE = PiecewiseLinearEncoding(bins)
    Periodic = PeriodicEmbeddings(num_frequencies=16)
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    att_tcr_to_epi = MultiHeadAttention(num_heads=hparams['num_heads'], key_dim=embed_dim)(
        query=tcr_input, value=epitope_input
    )
    pool_tcr = GlobalAveragePooling1D()(att_tcr_to_epi)

    att_epi_to_tcr = MultiHeadAttention(num_heads=hparams['num_heads'], key_dim=embed_dim)(
        query=epitope_input, value=tcr_input
    )
    pool_epi = GlobalAveragePooling1D()(att_epi_to_tcr)

    x = Concatenate()([pool_tcr, pool_epi])

    if hparams['embed_numerical'] == "PLE":
        feature_embeddings = PLE(feature_input)
    else:
        feature_embeddings = Periodic(feature_input)
        
    feature_embeddings_flatten = Flatten()(feature_embeddings)
    x = Concatenate()([x, feature_embeddings_flatten])

    # Deep Classifier Head (Residual MLP)
    # Using a residual loop allows the network to be deeper without degradation
    for _ in range(hparams['num_layers']):
        residual = x
        
        # Dense Block
        x = Dense(hparams['ff_dim'], activation=hparams['activation'], kernel_regularizer=l2(hparams['l2_reg']))(x)
        x = Dropout(hparams['dropout_rate'])(x)
        x = Dense(hparams['ff_dim'], activation=hparams['activation'], kernel_regularizer=l2(hparams['l2_reg']))(x)
        x = Dropout(hparams['dropout_rate'])(x)
        
        # Match dimensions for residual connection if needed
        if residual.shape[-1] != x.shape[-1]:
            residual = Dense(x.shape[-1])(residual)
            
        x = Add()([x, residual])
        x = LayerNormalization()(x)

    output = Dense(1, activation="sigmoid", name="Binding_Output")(x)
    
    model = Model(inputs=[tcr_input, epitope_input, feature_input], outputs=output)
    return model
