from typing import Mapping, Protocol
import tensorflow as tf
from .meta import Field, DType, Tensor

def schema(**fields: Field) -> Mapping[str, Field]:
  """Create a type specification to read/write into TFRecords."""
  return fields

class Feature(Protocol):
  def SerializeToString(self) -> bytes:
    ...

def tf_dtype(type: DType) -> tf.DType:
  match type:
    case 'float':
      return tf.float32
    case 'int':
      return tf.int64
    case 'string':
      return tf.string

def feature(schema: Field) -> tf.io.FixedLenFeature:
  match schema:
    case Tensor(shape, dtype):
      return tf.io.FixedLenFeature(shape, tf_dtype(dtype))
    case 'string':
      return tf.io.FixedLenFeature([], tf.string)
    case 'float':
      return tf.io.FixedLenFeature([], tf.float32)
    case 'int':
      return tf.io.FixedLenFeature([], tf.int64)

class parse:

  def __init__(self, schema: Mapping[str, Field]):
    self.features = { name: feature(schema[name]) for name in schema }

  def sample(self, record: tf.Tensor | bytes):
    return tf.io.parse_single_example(record, self.features)
  
  def batch(self, record: tf.Tensor | bytes):
    return tf.io.parse_example(record, self.features)
  
def serialize_field(field: Field, tensor) -> Feature:
  match field:
    case Tensor(shape, dtype):
      assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
      assert dtype in repr(tensor.dtype), f"Expected dtype {dtype}, got {tensor.dtype}"
      match dtype:
        case 'float':
          return tf.train.Feature(float_list=tf.train.FloatList(value=tensor.numpy().flatten())) # type: ignore
        case 'int':
          return tf.train.Feature(int64_list=tf.train.Int64List(value=tensor.numpy().flatten()))
        case 'string':
          return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor.numpy().flatten()]))
    case 'string':
      assert tensor.dtype == tf.string, f"Expected dtype string, got {tensor.dtype}"
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor.numpy()])) # type: ignore
    case 'float':
      assert tensor.dtype == tf.float32, f"Expected dtype float32, got {tensor.dtype}"
      return tf.train.Feature(float_list=tf.train.FloatList(value=tensor.numpy().flatten())) # type: ignore
    case 'int':
      assert tensor.dtype == tf.int64, f"Expected dtype int64, got {tensor.dtype}"
      return tf.train.Feature(int64_list=tf.train.Int64List(value=tensor.numpy().flatten()))
  
def serialize(schema: Mapping[str, Field], **tensors) -> bytes:
  features = { name: serialize_field(schema[name], tensor) for name, tensor in tensors.items() }
  return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
