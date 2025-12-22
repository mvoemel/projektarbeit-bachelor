import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, GlobalAveragePooling1D, Dropout, Concatenate, Flatten, Dot, Reshape, Conv2D, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from layers import PiecewiseLinearEncoding, PeriodicEmbeddings
from transformer_block import transformer_block
import numpy as np

# using SGD optimizer
HYPER_PARAMETERS = {
    'batch_size': 16, 
    'learning_rate': 0.0019879809384288102, 
    'dropout_rate': 0.3316929188322714, 
    'l2_reg': 0.009956055660659066, 
    'ff_dim': 141, 
    'num_layers': 2, 
    'num_heads': 41, 
    'activation': 'leaky_relu', 
    'embed_numerical': 'PLE'
}

# using ADAM optimizer (better AUC)
# HYPER_PARAMETERS = {
#     'batch_size': 32, 
#     'learning_rate': 0.0009070418872442555, 
#     'dropout_rate': 0.07029984566604908, 
#     'l2_reg': 0.007058882123454644, 
#     'ff_dim': 116, 
#     'num_layers': 4, 
#     'num_heads': 33, 
#     'activation': 'tanh', 
#     'embed_numerical': 'PLE'
# }

def create_model(hparams, embed_dim=64, feature_dim=12):
    print("Building Model v6 (Symmetric Cross-Attention + Transformer Block + Interaction Map (2D CNN))")
    
    bins = np.linspace(0.0, 1.0, num=11)
    PLE = PiecewiseLinearEncoding(bins)
    Periodic = PeriodicEmbeddings(num_frequencies=16)
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    # Branch 1: Symmetric Transformer Blocks
    
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

    # Branch 2: Interaction Map (2D CNN)
    interaction_map = Dot(axes=(2, 2))([tcr_input, epitope_input])
    interaction_map = Reshape((26, 24, 1))(interaction_map)
    
    x_cnn = Conv2D(32, (3, 3), activation='relu', padding='valid')(interaction_map)
    x_cnn = Conv2D(64, (3, 3), activation='relu', padding='valid')(x_cnn)
    pool_cnn = GlobalMaxPooling2D()(x_cnn)

    # Branch 3: Physicochemical Features
    if hparams['embed_numerical'] == "PLE":
        feature_embeddings = PLE(feature_input)
    else:
        feature_embeddings = Periodic(feature_input)
    feature_embeddings_flatten = Flatten()(feature_embeddings)

    x = Concatenate()([pool_tcr, pool_epi, pool_cnn, feature_embeddings_flatten])

    for _ in range(hparams['num_layers']):
        x = Dense(hparams['ff_dim'], activation=hparams['activation'], kernel_regularizer=l2(hparams['l2_reg']))(x)
        x = Dropout(hparams['dropout_rate'])(x)

    output = Dense(1, activation="sigmoid", name="Binding_Output")(x)
    
    model = Model(inputs=[tcr_input, epitope_input, feature_input], outputs=output)
    return model
