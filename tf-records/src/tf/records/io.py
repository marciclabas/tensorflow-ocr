from typing import Iterable, Literal, Mapping, Callable
import tensorflow as tf
import tf.records as tfr

def write(records: Iterable[bytes], filepath: str, *, compression: Literal['GZIP', 'ZLIB'] | None = None):
  """Write `records` to a TFRecord file."""
  opt = tf.io.TFRecordOptions(compression_type=compression or '')
  writer =  tf.io.TFRecordWriter(filepath, options=opt)
  for record in records:
    writer.write(record) # type: ignore
  writer.close()

def read(
  schema: Mapping[str, tfr.Field], filepaths: Iterable[str], *,
  compression: Literal['GZIP', 'ZLIB'] | None = None,
  deterministic: bool = True
) -> tf.data.Dataset:
  """Parse a series of TFRecord files into a single dataset"""
  ignore_order = tf.data.Options()
  ignore_order.deterministic = deterministic
  return (
    tf.data.TFRecordDataset(filepaths, compression_type=compression, num_parallel_reads=tf.data.AUTOTUNE)
    .with_options(ignore_order)
    .map(tfr.parse(schema).sample, num_parallel_calls=tf.data.AUTOTUNE)
  )

def batched_read(
  schema: Mapping[str, tfr.Field], filepaths: Iterable[str], *,
  compression: Literal['GZIP', 'ZLIB'] | None = None,
  deterministic: bool = True, batch_size: int = 32
) -> tf.data.Dataset:
  """Parse a series of TFRecord files into a single dataset"""
  ignore_order = tf.data.Options()
  ignore_order.deterministic = deterministic
  return (
    tf.data.TFRecordDataset(filepaths, compression_type=compression, num_parallel_reads=tf.data.AUTOTUNE)
    .with_options(ignore_order)
    .batch(batch_size)
    .map(tfr.parse(schema).batch, num_parallel_calls=tf.data.AUTOTUNE)
  )
