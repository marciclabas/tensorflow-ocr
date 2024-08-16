from typing import Literal, TypeAlias, Sequence, Mapping
from dataclasses import dataclass
from pydantic import BaseModel, Field as PyField

DType: TypeAlias = Literal['float', 'int', 'string']

@dataclass
class Tensor:
  """A dense tensor with fixed shape"""
  shape: Sequence[int]
  dtype: DType

@dataclass
class SeqTensor:
  """A dense tensor with variable first axis"""
  shape: Sequence[int]
  """Shape without the first axis, which is set to `None`"""
  dtype: DType
  tag: Literal['sequence'] = 'sequence'

Field: TypeAlias = Tensor | SeqTensor | DType

class Meta(BaseModel):
  files: str | Sequence[str] = '*.tfrecord.gz'
  compression: Literal['GZIP', 'ZLIB'] | None = None
  schema_: Mapping[str, Field] = PyField(alias='schema')
  num_samples: int | None = None

class MetaJson(BaseModel):
  tfrecords_dataset: Meta