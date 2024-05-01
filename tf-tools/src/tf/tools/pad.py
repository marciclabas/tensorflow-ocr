import tensorflow as tf
from jaxtyping import Int
from .typing import tf_function
from .set import set1d

@tf_function
def pad_to(x: tf.Tensor, target_shape: Int[tf.Tensor, "rank"]) -> tf.Tensor:
  """Pad `x` to `tf.maximum(target_shape, tf.shape(x))` (i.e., w/o truncating)"""
  shape = tf.shape(x)
  pads_end = tf.maximum(target_shape-shape, 0)
  pads = tf.stack([tf.zeros_like(pads_end), pads_end], axis=1)
  return tf.pad(x, pads)


@tf_function
def pad_dim_to(x: tf.Tensor, axis: int, length: int) -> tf.Tensor:
  """Pad `x`'s, `axis` dimension to `length`
  - Supports negative axes
  - Does nothing if `length < tf.shape(x)[axis]`"""
  target = set1d(tf.shape(x), axis, length)
  return pad_to(x, target)
