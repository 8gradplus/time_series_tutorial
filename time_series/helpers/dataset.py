import tensorflow as tf
from tensorflow.data import Dataset


def one_shot_iterator(ds: Dataset):
    return iter(ds.unbatch().map(lambda x, y: (tf.expand_dims(x, 0), y)))


def to_array(ds: Dataset):
    return tf.concat([batch for batch in ds], axis=0).numpy()


def extract_features(ds: Dataset):
    return ds.map(lambda x, y: x)


def extract_labels(ds: Dataset):
    return ds.map(lambda x, y: y)

