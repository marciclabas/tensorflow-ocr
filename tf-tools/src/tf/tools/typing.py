from functools import wraps
import tensorflow as tf
from jaxtyping import AbstractDtype

class Bytes(AbstractDtype):
  dtypes = ['object'] # yeah... apparently jaxtyping tests the numpy type

def tf_function(f, *args, **kwargs):
  """Like `tf.function` but uses `functools.wraps` to preserve linting of types and docstring
  - Use `functools.partial` to pass parameters
  """
  return wraps(f)(tf.function(f, *args, **kwargs))