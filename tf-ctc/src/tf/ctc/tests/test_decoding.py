from hypothesis import given, strategies as st, settings
import tensorflow as tf
from tf.ctc import beam_decode, greedy_decode, onehot_logits

def _test_exact(lab: list[int], decode):
  """Logits generated with `onehot_logits` should trivially decode to `lab`"""
  vlab = tf.sparse.from_dense([lab])
  logits = onehot_logits(vlab)
  paths, _ = decode(logits)
  dense_lab = tf.cast(tf.sparse.to_dense(vlab), tf.int32)
  dense_pred = tf.cast(tf.sparse.to_dense(paths[0]), tf.int32)
  assert tf.reduce_all(tf.equal(dense_lab, dense_pred))

@settings(deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=64))
def test_exact_beam(lab: list[int]):
  _test_exact(lab, beam_decode)

@settings(deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=64))
def test_exact_greedy(lab: list[int]):
  _test_exact(lab, greedy_decode)
