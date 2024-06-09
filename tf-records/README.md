# TFRecords

> Simple standard + tools for I/O of TFRecords

```
pip install tensorflow-records
```

## I/O

```python
import tensorflow as tf
import tf.records as tfr

image = tf.ones([1024, 768, 3], dtype=tf.uint8)
label = tf.constant('cat')

spec = tfr.spec(
  image=tfr.Tensor([1024, 768, 3], dtype='int'),
  label='string'
)

serialized = tfr.serialize(spec, image=x, label=y)
# b'\n+\n\x10\n\x05label\x12\x07\n\x05\n\x03cat\n\x17\n\x05image\x1...'

tfr.parse(spec).sample(tf.constant(serialized))
# { 'image': <tf.Tensor: shape=(1024, 768, 3) ...>, 'label': <tf.Tensor: shape=() ...> }


# or you can parse multiple at once
tfr.parse(spec).batch(tf.constant([serialized, serialized]))
# { 'image': <tf.Tensor: shape=(2, 1024, 768, 3) ...>, 'label': <tf.Tensor: shape=(2,) ...> }
```