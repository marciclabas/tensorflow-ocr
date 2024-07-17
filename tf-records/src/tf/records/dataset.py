from typing import Sequence, TextIO, Literal
from dataclasses import dataclass
import tf.tools as tft
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
  
  def iterate(self, *, mode: Literal['ordered', 'deterministic', 'shuffle'] | None = 'ordered', batch_size: int | None = None):
    """Read the dataset as a tf.data.Dataset
    - `mode == 'ordered'`: read the dataset in the order of the files (causes a signicant performance degradation when reading from multiple files)
    - `mode == 'deterministic'` (default): read the dataset in always the same order (passing `deterministic=True` to `tf.data.TFRecordDataset`)
    - `mode == 'shuffle'`: shuffle the files before reading (passing `deterministic=False` to `tf.data.TFRecordDataset`)
    - `mode == None`: pass `deterministic=False` but don't shuffle the files
    - `batch_size`: if provided, read the dataset in batches (often provides a significant speedup, above 2x)
    """
    files = self.meta.files
    if isinstance(files, str):
      from glob import glob
      files = glob(f'{self.base_path}/{files}')
      if mode == 'shuffle':
        import random
        random.shuffle(files)
      else:
        files.sort()
    else:
      files = [f'{self.base_path}/{f}' for f in files]

    deterministic = mode in ('deterministic', 'ordered')
    def read(files):
      if batch_size is None:
        return tfr.read(
          self.meta.schema_, files,
          compression=self.meta.compression, deterministic=deterministic
        )
      else:
        return tfr.batched_read(
          self.meta.schema_, files,
          compression=self.meta.compression, deterministic=deterministic, batch_size=batch_size
        )
    if mode == 'ordered':
      return tft.data.concat([read([f]) for f in files])
    else:
      return read(files)
  
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

def concat(datasets: Sequence[Dataset], *, mode: Literal['ordered', 'shuffle', 'deterministic'] | None = 'deterministic', batch_size: int | None = None):
  """Concatenate multiple datasets into a single one."""
  return tft.data.concat([ds.iterate(mode=mode, batch_size=batch_size) for ds in datasets])

def interleave(datasets: Sequence[Dataset], *, block_length: int = 1, mode: Literal['ordered', 'shuffle', 'deterministic'] | None = 'deterministic', batch_size: int | None = None):
  """Interleave multiple datasets into a single one."""
  return tft.data.interleave([ds.iterate(mode=mode, batch_size=batch_size) for ds in datasets], block_length=block_length)

def len(datasets: Sequence[Dataset]) -> int | None:
  """Total length of `keys` in all datasets. (Count as 0 if undefined)"""
  return sum((l for ds in datasets if (l := ds.len()) is not None))