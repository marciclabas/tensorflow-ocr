import tensorflow as tf
from jaxtyping import Shaped

def set1d(x: Shaped[tf.Tensor, "n"], idx: int, value) -> Shaped[tf.Tensor, "n"]:
  """Equivalent to `y = copy(x); y[idx] = value; return y`
  - Supports negative axes
  """
  axis = idx if idx >= 0 else tf.shape(x)[0] + idx
  return tf.tensor_scatter_nd_update(x, [[axis]], [value])