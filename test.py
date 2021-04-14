import tensorflow as tf
import torch

from tf_model_3 import SeqClassifier

embeddings = tf.convert_to_tensor(torch.load("./cache/slot/embeddings.pt"))
sample_transformer = SeqClassifier(
    mode='slot',
    text_len=35,
    embeddings=embeddings,
    batch_size=128,
    dropout=0.1,
    hidden_size=1024,
    num_layers=4,
    bidirectional=True,
    num_class=9
)

temp_input = tf.random.uniform((64, 35), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 35), dtype=tf.int64, minval=0, maxval=200)

fn_out = sample_transformer(temp_input, temp_target, training=False, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None)

print(fn_out)