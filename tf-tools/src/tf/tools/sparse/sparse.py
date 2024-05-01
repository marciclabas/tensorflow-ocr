import tensorflow as tf
from jaxtyping import Shaped, Bool
from ..typing import tf_function
from ..pad import pad_dim_to

@tf_function
def equal(xs: Shaped[tf.SparseTensor, "batch _"], ys: Shaped[tf.SparseTensor, "batch _"]) -> Bool[tf.Tensor, "batch"]:
  """Sample-wise equality, padding corresponding samples to the max length of the two"""
  dense_xs = tf.sparse.to_dense(xs)
  dense_ys = tf.sparse.to_dense(ys)
  maxlen = tf.maximum(tf.shape(dense_xs)[-1], tf.shape(dense_ys)[-1])
  padded_xs = pad_dim_to(dense_xs, axis=-1, length=maxlen)
  padded_ys = pad_dim_to(dense_ys, axis=-1, length=maxlen)
  return tf.reduce_all(tf.equal(padded_xs, padded_ys), axis=-1)

@tf_function
def any_equal(x: Shaped[tf.SparseTensor, "batch _"], ys: list[Shaped[tf.SparseTensor, "batch _"]]) -> Bool[tf.Tensor, "batch"]:
  """`any(equal(x, y) for y in ys)`, across the batch dimension"""
  return tf.reduce_any([equal(x, y) for y in ys], axis=0)
