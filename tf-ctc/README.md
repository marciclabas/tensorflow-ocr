# Tensorflow CTC

> Tools to simplify CTC in `tensorflow`

## Mocked Logits

'Perfect' logits to match any given labels. Basically, a large negative value ($10^{-9}$ by default) everywhere except $0$ at the appropriate index. Plus, blank characters (by default at index 0) interspersed to prevent collapse of equal, consecutive labels

```python
import tensorflow as tf
import tf.ctc as ctc

labs = tf.sparse.from_dense([[1, 2, 3, 0, 0],
                             [4, 3, 1, 2, 0],
                             [3, 1, 0, 0, 0]])

logits = ctc.onehot_logits(labs) # 'perfect' logits

# <tf.Tensor: shape=(3, 10, 5), dtype=float32, numpy=
#
# array([[[-1.e+09,  0.e+00, -1.e+09, -1.e+09, -1.e+09],
#         [ 0.e+00, -1.e+09, -1.e+09, -1.e+09, -1.e+09],
#         [-1.e+09, -1.e+09,  0.e+00, -1.e+09, -1.e+09],
#         [ 0.e+00, -1.e+09, -1.e+09, -1.e+09, -1.e+09],
#         [-1.e+09, -1.e+09, -1.e+09,  0.e+00, -1.e+09],
#         [ 0.e+00, -1.e+09, -1.e+09, -1.e+09, -1.e+09],
#         [ 0.e+00, -1.e+09, -1.e+09, -1.e+09, -1.e+09],
#         [ 0.e+00, -1.e+09, -1.e+09, -1.e+09, -1.e+09],
#         [ 0.e+00, -1.e+09, -1.e+09, -1.e+09, -1.e+09],
#         [ 0.e+00, -1.e+09, -1.e+09, -1.e+09, -1.e+09]],
# 
#        [[-1.e+09, -1.e+09, -1.e+09, -1.e+09,  0.e+00],
#           ...
#         [ 0.e+00, -1.e+09, -1.e+09, -1.e+09, -1.e+09]],
# 
#        [[-1.e+09, -1.e+09, -1.e+09,  0.e+00, -1.e+09],
#           ...
#         [ 0.e+00, -1.e+09, -1.e+09, -1.e+09, -1.e+09]]], dtype=float32)>

ctc.loss(labs, logits) # something very close to [0, 0, 0]
```

## Loss

Wrapper around `tf.nn.loss` but labels must be a `SparseTensor` (as they should probably be) and the API is trivial (just labels and logits)

(See example above)

## Decoding

Wrappers around `tf.nn.ctc_greedy_decoder` and `tf.nn.ctc_beam_search_decoder`, except:
- Beam Search supports setting the blank index to 0 (and does so by default)
- Logits are batch-major (as you most likely already have them)
- Trivial API (just pass the logits and optionally config)

```python
import tensorflow as tf
import tf_ctc as ctc

labs = tf.sparse.from_dense([[1, 2, 3, 0, 0],
                             [4, 3, 1, 2, 0],
                             [3, 1, 0, 0, 0]])

logits = ctc.onehot_logits(labs) # 'perfect' logits

[top_path, *_], log_probs = ctc.beam_decode(logits)
# or
[top_path], log_probs = ctc.greedy_decode(logits)

tf.sparse.to_dense(top_path)
# <tf.Tensor: shape=(3, 4), dtype=int64, numpy=
# array([[1, 2, 3, 0],
#        [4, 3, 1, 2],
#        [3, 1, 0, 0]])>
```

## Metrics

Generalizations of *accuracy* and *edit distance* (default to $k=1$, i.e. the usual, concrete versions):
- **Top-$k$ accuracy**: proportion of samples where the label is in the top-$k$ predictions
- **Top-$k$ edit distance**: minimum edit distance between the label and each of the top-$k$ predictions; averaged across all samples
  
```python
import tensorflow as tf
import tf_ctc as ctc

labs = tf.sparse.from_dense([[1, 2, 3, 0, 0],
                             [4, 3, 1, 2, 0],
                             [3, 1, 0, 0, 0]])

logits = ctc.onehot_logits(labs) # 'perfect' logits

ctc.accuracy(labs, logits) # 1.0
ctc.edit_distance(labs, logits) # 0.0
```

## Testing

Very simple randomized tests using `ctc.onehot_logits`

```bash
pip install tf-ctc[test]
python -m tf.ctc.test
```

(Note: tests may spit out a warning about `jaxtyping`. That's just fine)