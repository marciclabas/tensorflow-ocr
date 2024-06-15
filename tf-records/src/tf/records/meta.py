from typing import Literal, TypeAlias, Sequence, Mapping
from dataclasses import dataclass
from pydantic import BaseModel, Field as PyField

DType: TypeAlias = Literal['float', 'int', 'string']

@dataclass
class Tensor:
  shape: Sequence[int]
  dtype: DType

Field: TypeAlias = Tensor | DType

class Meta(BaseModel):
  files: str | Sequence[str] = '*.tfrecord.gz'
  compression: Literal['GZIP', 'ZLIB'] | None = None
  schema_: Mapping[str, Field] = PyField(alias='schema')
  num_samples: int | None = None

class MetaJson(BaseModel):
  tfrecords_dataset: Meta