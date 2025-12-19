import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, GlobalAveragePooling1D, Dropout, Concatenate, Flatten, Dot, Conv2D, GlobalMaxPooling2D, Reshape, Add, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from layers import PiecewiseLinearEncoding, PeriodicEmbeddings
import numpy as np

# using SGD optimizer
HYPER_PARAMETERS = {
    'batch_size': 64, 
    'learning_rate': 0.005285426279175433, 
    'dropout_rate': 0.46594630836578194, 
    'l2_reg': 4.047725053283987e-05, 
    'ff_dim': 123, 
    'num_layers': 2, 
    'num_heads': 12, 
    'activation': 'tanh', 
    'embed_numerical': 'Periodic'
}

# using ADAM optimizer
HYPER_PARAMETERS = {
    'batch_size': 32, 
    'learning_rate': 0.0003132071576424828, 
    'dropout_rate': 0.00396683619911593, 
    'l2_reg': 1.8144367529691835e-05, 
    'ff_dim': 47, 
    'num_layers': 4, 
    'num_heads': 22, 
    'activation': 'tanh', 
    'embed_numerical': 'PLE'
}

def create_model(hparams, embed_dim=64, feature_dim=12):
    print("Building Model v5 (Symmetric Cross-Attention + Interaction Map (2D CNN) + Deep ResNet Classifier Head)")
    
    bins = np.linspace(0.0, 1.0, num=11)
    PLE = PiecewiseLinearEncoding(bins)
    Periodic = PeriodicEmbeddings(num_frequencies=16)
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    att_tcr_to_epi = MultiHeadAttention(num_heads=hparams['num_heads'], key_dim=embed_dim)(
        query=tcr_input, value=epitope_input
    )
    pool_tcr_to_epi = GlobalAveragePooling1D()(att_tcr_to_epi)

    att_epi_to_tcr = MultiHeadAttention(num_heads=hparams['num_heads'], key_dim=embed_dim)(
        query=epitope_input, value=tcr_input
    )
    pool_epi_to_tcr = GlobalAveragePooling1D()(att_epi_to_tcr)

    interaction_map = Dot(axes=(2, 2))([tcr_input, epitope_input])
    interaction_map = Reshape((26, 24, 1))(interaction_map)
    
    x_cnn = Conv2D(32, (3, 3), activation='relu', padding='valid')(interaction_map)
    x_cnn = Conv2D(64, (3, 3), activation='relu', padding='valid')(x_cnn)
    pool_cnn = GlobalMaxPooling2D()(x_cnn)

    if hparams['embed_numerical'] == "PLE":
        feature_embeddings = PLE(feature_input)
    else:
        feature_embeddings = Periodic(feature_input)
    feature_embeddings_flatten = Flatten()(feature_embeddings)

    x = Concatenate()([pool_tcr_to_epi, pool_epi_to_tcr, pool_cnn, feature_embeddings_flatten])

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
