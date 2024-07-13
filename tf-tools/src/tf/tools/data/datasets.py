from typing import Sequence
import tensorflow as tf

def interleave(datasets: Sequence[tf.data.Dataset], *, block_length: int = 1) -> tf.data.Dataset:
  return tf.data.Dataset.from_tensor_slices(datasets).interleave(
    lambda x: x,
    cycle_length=len(datasets),
    block_length=block_length
  )

def concat(datasets: Sequence[tf.data.Dataset]) -> tf.data.Dataset:
  return tf.data.Dataset.from_tensor_slices(datasets).flat_map(lambda x: x)