def shard_size(
  num_samples: int, sample_bytes: float, *,
  min_shard_bytes: int = 100*1024*1024,
  min_shards: int = 10
):
  """Compute the shard split. `min_shard_bytes` takes priority over `min_shards`.
  Default values taken from the [tensorflow docs](https://www.tensorflow.org/tutorials/load_data/tfrecord#:~:text=data%20API%20for%20dataset%20performance,will%20be%20hosts%20reading%20data.)
  """
  total_size = num_samples * sample_bytes
  avg_size = total_size / min_shards
  return max(avg_size, min_shard_bytes)
