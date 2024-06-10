from typing import Iterable, TextIO
from dataclasses import dataclass
from haskellian import Iter
import tf.records as tfr
from .meta import Meta, MetaJson

@dataclass
class Dataset:
  meta: Meta
  base_path: str

  @classmethod
  def read(cls, path: str) -> 'Dataset':
    with open(f'{path}/meta.json') as f:
      meta = MetaJson.model_validate_json(f.read()).tfrecords_dataset
    return cls(meta, path)
  
  def iterate(self, *, keep_order: bool = True, batch_size: int | None = None):
    """Read the dataset as a tf.data.Dataset
    - `batch_size`: if provided, read the dataset in batches (often provides a significant speedup, above 2x)
    """
    files = self.meta.files
    if isinstance(files, str):
      from glob import glob
      files = glob(f'{self.base_path}/{files}')

    if batch_size is None:
      return tfr.read(
        self.meta.schema_, files,
        compression=self.meta.compression, keep_order=keep_order
      )
    else:
      return tfr.batched_read(
        self.meta.schema_, files,
        compression=self.meta.compression, keep_order=keep_order, batch_size=batch_size
      )
  
  def len(self) -> int | None:
    return self.meta.num_samples
  
def glob(glob: str, *, recursive: bool = False, err_stream: TextIO | None = None) -> list[Dataset]:
  """Read all datasets that match a glob pattern."""
  from glob import glob as _glob
  datasets = []
  for p in sorted(_glob(glob, recursive=recursive)):
    try:
      datasets.append(Dataset.read(p))
    except Exception as e:
      if err_stream:
        print(f'Error reading dataset at {p}:', e, file=err_stream)
  return datasets

def chain(datasets: Iterable[Dataset], *, keep_order: bool = True):
  """Chain multiple datasets into a single one."""
  import tensorflow as tf
  ds = Iter(datasets).map(lambda ds: ds.iterate(keep_order=keep_order)).reduce(tf.data.Dataset.concatenate)
  return ds or tf.data.Dataset.from_tensors({})

def len(datasets: Iterable[Dataset]) -> int | None:
  """Total length of `keys` in all datasets. (Count as 0 if undefined)"""
  return sum((l for ds in datasets if (l := ds.len()) is not None))