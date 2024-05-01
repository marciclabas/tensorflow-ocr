from hypothesis import given, strategies as st, settings
import tensorflow as tf
import keras
import numpy as np
from tf.ctc import edit_distance, accuracy, onehot_logits

@settings(deadline=None)
@given(st.lists(st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=64), min_size=1, max_size=8))
def test_perfect_ed(labs: list[list[int]]):
  batch = tf.sparse.from_dense(keras.utils.pad_sequences(labs))
  logits = onehot_logits(batch)
  assert tf.reduce_all(edit_distance(batch, logits) == 0)

@settings(deadline=None)
@given(st.lists(st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=64), min_size=1, max_size=8))
def test_perfect_acc(labs: list[list[int]]):
  batch = tf.sparse.from_dense(keras.utils.pad_sequences(labs, padding='post', truncating='post'))
  logits = onehot_logits(batch)
  assert np.isclose(accuracy(batch, logits), 1)