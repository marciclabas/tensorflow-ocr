from hypothesis import given, strategies as st, settings
import tensorflow as tf
import numpy as np
from tf.ctc import loss, onehot_logits

@settings(deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=64), st.integers(min_value=0, max_value=12))
def test_zero(lab: list[int], delta: int):
  min_depth = tf.reduce_max(lab)+1
  depth = min_depth + delta
  vlab = tf.sparse.from_dense([lab])
  logits = onehot_logits(vlab, vocabsize=depth)
  assert np.isclose(loss(vlab, logits), 0)