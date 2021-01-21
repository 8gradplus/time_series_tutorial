from pandas import DataFrame
import tensorflow as tf
from tensorflow.data import Dataset


class Offsets:
    """Define relevant offsets for time series"""
    def __init__(self,
                 input_width: int,
                 label_width: int,
                 shift: int = None):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        if not shift:
            self.shift = self.label_width
        self.sequence_width = self.input_width + self.shift
        self.check()

    def check(self):
        msg = "Label width needs to be smaller than offset in order to make future predictions"
        assert self.label_width <= self.shift, msg


class MakeDatasetFromDataFrame:
    """Generate tensorflow dataset for time series from pandas dataframe"""

    def __init__(self,
                 offsets: Offsets,
                 batch_size: int,
                 labels: list,
                 shuffle_buffer_size: int,
                 prefetch: int = 1):
        self.offsets = offsets
        self.batch_size = batch_size
        self.labels = labels
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch = prefetch

    def __call__(self, df: DataFrame):
        sequence_width = self.offsets.sequence_width
        label_indices = get_label_indices(df, self.labels)
        xs = (Dataset.from_tensor_slices(df)
                     .window(sequence_width, 1, drop_remainder=True)
                     .flat_map(BatchWindow(sequence_width))
                     .map(SplitInTime(self.offsets))
                     .map(PickLabels(label_indices))
                     .batch(self.batch_size))
        if self.shuffle_buffer_size:
            xs = xs.shuffle(self.shuffle_buffer_size)
        if self.prefetch:
            xs = xs.prefetch(1)
        return xs


class SplitInTime:
    """Split time window in features and targets according to offsets"""

    def __init__(self, offsets: Offsets):
        self.offsets = offsets

    def __call__(self, t: tf.Tensor):
        features = t[: -self.offsets.shift]
        labels = t[-self.offsets.label_width:]
        return features, labels


def get_label_indices(df: DataFrame, labels: list):
    """Get column indices of specified labels of pandas dataframe"""
    return [idx for idx, name in enumerate(df.columns) if name in labels]


class PickLabels:
    """Pick only those columns of tensor as labels which are spezified by label indices"""

    def __init__(self, label_indices: list):
        self.label_indices = label_indices

    def __call__(self, features, labels):
        # tf is less flexible than numpy. so we take each tensor slice individually and stack it then
        picked_labels = [labels[:, i] for i in self.label_indices]
        picked_labels = tf.stack(picked_labels, axis=1)
        return features, picked_labels


class BatchWindow:
    """Batch ds according to sequence width"""

    def __init__(self, sequence_width: int):
        self.sequence_width = sequence_width

    def __call__(self, window: Dataset):
        return window.batch(self.sequence_width)


