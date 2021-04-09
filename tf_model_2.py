''' Libraries '''
import math
from typing import Dict

import numpy as np
import tensorflow as tf


''' Function '''
class PositionalEncoding300k(tf.keras.layers.Layer):
    def __init__(self, batch_size, text_len, dropout=0.1):
        super(PositionalEncoding300k, self).__init__()
        dim = 300
        posit = np.arange(text_len)[:, np.newaxis]                       # (28, 1)
        depth = np.arange(dim)[np.newaxis, :]                            # (1, 300)
        angle = posit / np.power(10000, (2*(depth//2)/np.float32(dim)))  # (28, 300)
        angle[:, 0::2] = np.sin(angle[:, 0::2])
        angle[:, 1::2] = np.cos(angle[:, 1::2])

        self.pos_enc = tf.convert_to_tensor(
            np.concatenate([angle[np.newaxis, :, :]]*batch_size),
            dtype=tf.float32
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        x = x + self.pos_enc[:tf.shape(x)[0], :, :]
        return self.dropout(x)

    def get_config(self):
        return {
            'pos_enc': self.pos_enc,
            'dropout': self.dropout
        }


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads  = num_heads
        self.dim = dim

        assert dim % self.num_heads == 0
        self.depth = dim // self.num_heads

        self.W_q = tf.keras.layers.Dense(dim)
        self.W_k = tf.keras.layers.Dense(dim)
        self.W_v = tf.keras.layers.Dense(dim)

        self.dense = tf.keras.layers.Dense(dim)

    def split_heads(self, x):
        # print(f"x.shape:                 {x.shape}   <-- should be (batch_size, text_len, dim)")
        x = tf.reshape(x, (tf.shape(x)[0], x.shape[1], self.num_heads, self.depth))
        # print(f"x.shape:                 {x.shape}   <-- should be (batch_size, text_len, num_heads, depth)")
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        # print(f"x.shape:                 {x.shape}   <-- should be (batch_size, num_heads, text_len, depth)")
        return x

    def scaled_dot_product_attention(self, Q, K, V, mask):
        QK = tf.matmul(Q, K, transpose_b=True)
        # print(f"QK.shape:                {QK.shape}   <-- should be (batch_size, num_heads, text_len, text_len)")
        dim_k = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = QK / tf.math.sqrt(dim_k)
        if mask is not None: scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # print(f"attention_weights.shape: {attention_weights.shape}   <-- should be (batch_size, num_heads, text_len, text_len)")
        output = tf.matmul(attention_weights, V)
        # print(f"output.shape:            {output.shape}   <-- should be (batch_size, num_heads, text_len, depth)")
        return output

    def call(self, Q, K, V, mask):
        # print(f"Q.shape:                 {Q.shape}   <-- should be (batch_size, text_len, dim)")
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        # print(f"Q.shape:                 {Q.shape}   <-- should be (batch_size, text_len, dim)")
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        # print(f"Q.shape:                 {Q.shape}   <-- should be (batch_size, num_heads, text_len, depth)")
        scaled_attention = self.scaled_dot_product_attention(Q, K, V, mask)
        # print(f"scaled_attention.shape:  {scaled_attention.shape}   <-- should be (batch_size, num_heads, text_len, depth)")
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # print(f"scaled_attention.shape:  {scaled_attention.shape}   <-- should be (batch_size, text_len, num_heads, depth)")
        concat_attention = tf.reshape(scaled_attention,
            shape=(tf.shape(scaled_attention)[0], scaled_attention.shape[1], self.dim)
        )
        # print(f"concat_attention.shape:  {concat_attention.shape}   <-- should be (batch_size, text_len, dim)")
        output = self.dense(concat_attention)
        # print(f"output.shape:            {output.shape}   <-- should be (batch_size, text_len, dim)")
        return output


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, hidden_size, dropout):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(dim, num_heads)
        self.mha_dropout = tf.keras.layers.Dropout(dropout)
        self.mha_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Pointwise Feed Forward network
        self.pff = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu),
            tf.keras.layers.Dense(dim),
        ])
        self.pff_dropout = tf.keras.layers.Dropout(dropout)
        self.pff_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        x += self.mha_dropout(attn_output, training=training)
        x = self.mha_layer_norm(x)

        pff_output = self.pff(x)
        x += self.pff_dropout(pff_output, training=training)
        x = self.pff_layer_norm(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, hidden_size, dropout, num_layers, bidirectional):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.encoder_layers_forward = [ EncoderLayer(
            dim=300,
            num_heads=12,
            hidden_size=hidden_size,
            dropout=dropout
        ) for _ in range(num_layers) ]
        if bidirectional:
            self.encoder_layers_backward = [ EncoderLayer(
                dim=300,
                num_heads=12,
                hidden_size=hidden_size,
                dropout=dropout
            ) for _ in range(num_layers) ]
            self.dense = tf.keras.layers.Dense(300)
    
    def call(self, x, training, mask):
        x_forward = x
        for encoder_layer in self.encoder_layers_forward:
            x_forward = encoder_layer(x_forward, mask=mask)
        if self.bidirectional:
            x_backward = x
            x_backward = tf.reverse(x_backward, axis=[-2])
            for encoder_layer in self.encoder_layers_backward:
                x_backward = encoder_layer(x_backward, mask=mask)
            x_backward = tf.reverse(x_backward, axis=[-2])
            x = tf.concat([x_forward, x_backward], axis=-1)
            x = self.dense(x)
        else:
            x = x_forward
        return x


class SeqClassifier(tf.keras.Model):
    def __init__(
        self,
        embeddings: tf.Tensor,
        text_len: int,    # Added by me
        batch_size: int,  # Added by me
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        mode: str,        # Added by me
    ):
        super(SeqClassifier, self).__init__()
        # TODO: model architecture

        self.mode = mode
        # (BATCH_SIZE, text_len)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=6491,
            output_dim=300,
            input_length=text_len,
            embeddings_initializer=tf.constant_initializer(embeddings.numpy())
        )
        # (BATCH_SIZE, text_len, 300)
        self.pos_enc_300k = PositionalEncoding300k(
            text_len=text_len,
            batch_size=batch_size
        )
        # (BATCH_SIZE, text_len, 300)
        self.encoder = Encoder(
            dim=300,
            num_heads=12,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            bidirectional=bidirectional
        )
        # (BATCH_SIZE, text_len, 300)
        if mode == 'intent':
            self.droupout_1 = tf.keras.layers.Dropout(dropout)
            self.last_layers_1 = [
                # (BATCH_SIZE, text_len, 300)
                tf.keras.layers.Dense(450),
                # (BATCH_SIZE, text_len, 450)
                tf.keras.layers.Dense(300),
                # (BATCH_SIZE, text_len, 300)
                tf.keras.layers.Dense(100),
                # (BATCH_SIZE, text_len, 100)
                tf.keras.layers.Dense(25),
                # (BATCH_SIZE, text_len, 25)
                tf.keras.layers.Dense(5),
                # (BATCH_SIZE, text_len, 5)
                tf.keras.layers.Dense(1),
                # (BATCH_SIZE, text_len, 1)
                tf.keras.layers.Reshape((text_len, )),
            ]
            self.droupout_2 = tf.keras.layers.Dropout(dropout)
            self.last_layers_2 = [
                # (BATCH_SIZE, text_len)
                tf.keras.layers.Dense(50),
                # (BATCH_SIZE, 50)
                tf.keras.layers.Dense(100),
                # (BATCH_SIZE, 100)
                tf.keras.layers.Dense(200),
                # (BATCH_SIZE, 200)
                tf.keras.layers.Dense(150, activation=tf.nn.softmax),
                # (BATCH_SIZE, 150)
            ]
        elif mode == 'slot':
            self.droupout = tf.keras.layers.Dropout(dropout)
            self.last_layers = [
                # (BATCH_SIZE, text_len, 300)
                tf.keras.layers.Dense(450),
                # (BATCH_SIZE, text_len, 450)
                tf.keras.layers.Dense(300),
                # (BATCH_SIZE, text_len, 300)
                tf.keras.layers.Dense(100),
                # (BATCH_SIZE, text_len, 100)
                tf.keras.layers.Dense(30),
                # (BATCH_SIZE, text_len, 30)
                tf.keras.layers.Dense(9, activation=tf.nn.softmax),
                # (BATCH_SIZE, text_len, 9)
            ]
        else: raise Exception

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def call(self, encoded_tokens, training) -> Dict[str, tf.Tensor]:
        x = self.embedding(encoded_tokens)
        x = self.pos_enc_300k(x)
        # x = self.encoder(x, mask=np.triu(np.ones([text_len, text_len]), k=1))
        x = self.encoder(x, mask=None)
        if self.mode == 'intent':
            x = self.droupout_1(x, training=training)
            for layer in self.last_layers_1: x = layer(x)
            x = self.droupout_2(x, training=training)
            for layer in self.last_layers_2: x = layer(x)
        elif self.mode == 'slot':
            x = self.droupout(x, training=training)
            for layer in self.last_layers: x = layer(x)
        return x