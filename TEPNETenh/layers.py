import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


class PiecewiseLinearEncoding(Layer):
    def __init__(self, bins, **kwargs):
        super(PiecewiseLinearEncoding, self).__init__(**kwargs)
        self.bins = tf.convert_to_tensor(bins, dtype=tf.float32)

    def call(self, inputs):
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        bin_widths = self.bins[1:] - self.bins[:-1]
        bin_edges = (inputs_expanded - self.bins[:-1]) / bin_widths
        bin_edges = tf.clip_by_value(bin_edges, 0.0, 1.0)
        return bin_edges

    def get_config(self):
        config = super().get_config()
        config.update({"bins": self.bins.numpy().tolist()})
        return config


class PeriodicEmbeddings(Layer):
    def __init__(self, num_frequencies=16, **kwargs):
        super(PeriodicEmbeddings, self).__init__(**kwargs)
        self.num_frequencies = num_frequencies
        self.freqs = tf.Variable(
            initial_value=tf.random.uniform(shape=(num_frequencies,), minval=0.1, maxval=1.0),
            trainable=True
        )

    def call(self, inputs):
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        periodic_features = tf.concat([
            tf.sin(2 * np.pi * inputs_expanded * self.freqs),
            tf.cos(2 * np.pi * inputs_expanded * self.freqs)
        ], axis=-1)
        return periodic_features

    def get_config(self):
        config = super().get_config()
        config.update({"num_frequencies": self.num_frequencies})
        return config
