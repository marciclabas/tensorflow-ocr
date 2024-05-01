import tensorflow as tf
from jaxtyping import Float, Int
from tf.tools import tf_function

@tf_function
def beam_decode_tf(
	logits: Float[tf.Tensor, "batch maxlen vocabsize"],
	beam_width: int = 100, top_paths: int = 1
) -> tuple[list[Int[tf.SparseTensor, "batch maxlen"]], Float[tf.Tensor, "batch 1"]]:
	"""CTC beam search decoding
	- `logits :: FloatTensor[batch, maxlen, vocabsize]`
	- Returns `(top_paths, log_probs)`
		- `top_paths :: list[IntSparseTensor[batch, maxlen]]` where elements correspond to indices in the vocabulary or 0 for blank (so, in the closed interval `[0, vocabsize]`)
		- `log_probs :: list[FloatTensor[batch, 1]]`
		
		Note that `len(top_paths) == len(log_probs) == top_paths`
	"""
	batch = tf.shape(logits)[0] # type: ignore
	maxlen = tf.shape(logits)[1] # type: ignore
	time_major_logits = tf.transpose(logits, perm=[1, 0, 2])
	top_preds, log_probs = tf.nn.ctc_beam_search_decoder(
		inputs=time_major_logits,
		sequence_length=tf.repeat(maxlen, batch),
		beam_width=beam_width, top_paths=top_paths,
	) # type: ignore
	return top_preds, log_probs

@tf_function
def shift_left(logits: tf.Tensor) -> tf.Tensor:
    """Shifts last dimension left by 1: `0 -> n-1, 1 -> 0, 2 -> 1, ...`"""
    vocab_size = tf.shape(logits)[-1] # type: ignore
    index_mapping = tf.concat([
        tf.range(1, vocab_size),  # Shift non-blank indices up by 1
        tf.constant([0], dtype=tf.int32)  # Move blank to the end
    ], axis=0)
    return tf.gather(logits, index_mapping, axis=-1)

@tf_function
def beam_decode(
	logits: Float[tf.Tensor, "batch maxlen vocabsize"],
	beam_width: int = 100, top_paths: int = 1, blank_zero: bool = True
) -> tuple[list[Int[tf.SparseTensor, "batch maxlen"]], Float[tf.Tensor, "batch 1"]]:
	"""
CTC beam search decoding, but blank index can be set to 0
- `logits :: FloatTensor[batch, maxlen, vocabsize]`
- `blank_zero`: if `True` (the default), the blank index is set to 0
- Returns `(top_paths, log_probs)`
- `top_paths :: list[IntSparseTensor[batch, maxlen]]` where elements correspond to indices in the vocabulary or 0 for blank (so, in the closed interval `[0, vocabsize]`)
- `log_probs :: list[FloatTensor[batch, 1]]`
- Note that `len(top_paths) == len(log_probs) == top_paths`


#### How does `blank_zero = True` work?
1. Shift the logit elements left by 1 in the last dimension (`0` goes to `n-1`, `1` to `0`, `2` to `1`, etc.)
2. Run the usual `tf.nn.ctc_beam_decoder`
3. Add 1 to the predicted indices (so, `0 -> 1, 1 -> 2, n-2 -> n-1`. `n-1` was blank thus there can be no `n-1` in the predicted sequences)
	"""
	z = shift_left(logits) if blank_zero else logits
	t, log_probs = beam_decode_tf(z, beam_width, top_paths)
	top_preds: list[tf.SparseTensor] = t # thank you python type linter!
	paths = [s.with_values(s.values+1) for s in top_preds] if blank_zero else top_preds # type: ignore
	return paths, log_probs

@tf_function
def greedy_decode(
    logits: Float[tf.Tensor, "batch maxlen vocabsize"]    
) -> tuple[list[Int[tf.SparseTensor, "batch maxlen"]], Float[tf.Tensor, "batch 1"]]:
	"""CTC greedy decoding
	- `logits :: FloatTensor[batch, maxlen, vocabsize]`
	- Returns `(top_paths, log_probs)`. Just like `beam_decode`, except these are both singleton lists
		- `top_paths[0] :: IntSparseTensor[batch, maxlen]` where elements correspond to indices in the vocabulary or 0 for blank (so, in the closed interval `[0, vocabsize]`)
		- `log_probs[0] :: FloatTensor[batch, 1]`
		
		Note that `len(top_paths) == len(log_probs) == top_paths`
	"""
	batch = tf.shape(logits)[0] # type: ignore
	maxlen = tf.shape(logits)[1] # type: ignore
	time_major_logits = tf.transpose(logits, perm=[1, 0, 2])
	top_paths, log_probs = tf.nn.ctc_greedy_decoder(
		inputs=time_major_logits,
		sequence_length=tf.repeat(maxlen, batch),
		blank_index=0
	)
	return top_paths, log_probs