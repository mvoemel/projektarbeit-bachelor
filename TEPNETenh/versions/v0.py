import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, GlobalAveragePooling1D, Dropout, Concatenate, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from layers import PiecewiseLinearEncoding, PeriodicEmbeddings
import numpy as np

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
    print("Building Model v0 (Baseline)")
    
    bins = np.linspace(0.0, 1.0, num=11)
    PLE = PiecewiseLinearEncoding(bins)
    Periodic = PeriodicEmbeddings(num_frequencies=16)
    
    tcr_input = Input(shape=(26, embed_dim), name="TCR_Input")
    epitope_input = Input(shape=(24, embed_dim), name="Epitope_Input")
    feature_input = Input(shape=(feature_dim,), name="Physicochemical_Features")
    
    attention_output = MultiHeadAttention(num_heads=hparams['num_heads'], key_dim=embed_dim)(
        query=tcr_input, value=epitope_input
    )
    x = GlobalAveragePooling1D()(attention_output)

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
