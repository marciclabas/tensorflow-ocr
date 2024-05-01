import tensorflow as tf
from jaxtyping import Float, Int
from tf.tools import tf_function

@tf_function
def loss(
  labels: Int[tf.SparseTensor, "batch maxlen"],
  logits: Float[tf.Tensor, "batch maxlen vocab"],
  blank_index: int = 0
) -> Float[tf.Tensor, "batch"]:
  """CTC loss. `blank_index` is the index in the vocabulary corresponding to the blank character"""
  batch = tf.shape(logits)[0]
  maxlen = tf.shape(logits)[1]
  logit_lens = tf.repeat(maxlen, batch)
  return tf.nn.ctc_loss(tf.cast(labels, tf.int32), logits, None, logit_lens, logits_time_major=False, blank_index=blank_index)