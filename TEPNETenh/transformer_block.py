import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Dense, Dropout, Add, LayerNormalization


def transformer_block(query, key_value, num_heads, embed_dim, ff_dim, dropout_rate):
    """
    Standard Transformer Block: Attention -> Add & Norm -> FFN -> Add & Norm
    """
    # 1. Multi-Head Attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        query=query, value=key_value
    )
    attn_output = Dropout(dropout_rate)(attn_output)
    
    # 2. Add & Norm (Residual Connection 1)
    # Note: We add query to the output. Shapes must match.
    out1 = LayerNormalization(epsilon=1e-6)(Add()([query, attn_output]))
    
    # 3. Feed Forward Network
    ffn_output = Dense(ff_dim, activation="relu")(out1) # Expand
    ffn_output = Dense(embed_dim)(ffn_output)           # Project back
    ffn_output = Dropout(dropout_rate)(ffn_output)
    
    # 4. Add & Norm (Residual Connection 2)
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ffn_output]))
    
    return out2
