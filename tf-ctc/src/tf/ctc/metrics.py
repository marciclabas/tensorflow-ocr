import tensorflow as tf
from jaxtyping import Int, Float
from .decoding import beam_decode
import tf.tools as tft

def accuracy(
  labs: Int[tf.SparseTensor, "batch maxlen1"],
  logits: Float[tf.Tensor, "batch maxlen2 vocabsize"],
  beam_width: int = 100, k: int = 1, blank_zero: bool = True
) -> float:
  """Top `k`-accuracy of the batch
  
  (i.e. proportion of samples that correctly predict the corresponding label in one of the top `k` paths)
  - `blank_zero`: if `True` (the default), the blank index is set to 0 (see `tf_ctc.beam_decode` for details)
  """
  top_paths, _ = beam_decode(logits, beam_width=beam_width, top_paths=k, blank_zero=blank_zero)
  labs64 = tf.cast(labs, tf.int64) # thank you tensorflow for always knowing when to enforce precision! xd
  sample_eqs = tft.sparse.any_equal(labs64, top_paths)
  return tf.reduce_mean(tf.cast(sample_eqs, tf.float32))

def edit_distance(
  labs: Int[tf.SparseTensor, "batch maxlen1"],
  logits: Float[tf.Tensor, "batch maxlen2 vocabsize"],
  beam_width: int = 100, k: int = 1, blank_zero: bool = True
) -> float:
  """#### Top `k` edit distance of the batch
  
  (i.e. lowest edit distance between the label and each of the top `k` predictions)
  - `blank_zero`: if `True` (the default), the blank index is set to 0 (see `tf_ctc.beam_decode` for details)
  """
  top_paths, _ = beam_decode(logits, beam_width=beam_width, top_paths=k, blank_zero=blank_zero)
  labs64 = tf.cast(labs, tf.int64) # thank you tensorflow for always knowing when to enforce precision! xd
  sample_eds = tf.reduce_min([tf.edit_distance(labs64, p) for p in top_paths], axis=0)
  return tf.reduce_mean(sample_eds)