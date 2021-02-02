from dataclasses import dataclass
from pandas import DataFrame
from logging import getLogger

logger = getLogger(__name__)


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
        msg = "Label width smaller than offset - not all labels are in the future"
        if self.label_width <= self.shift:
            logger.warning(msg)


@dataclass
class Data:
    train: DataFrame
    validation: DataFrame
    test: DataFrame
    column_indices: dict

    def map(self, f):
        return Data(f(self.train), f(self.validation), f(self.test), self.column_indices)

    def on_train(self, f):
        return Data(f(self.train), self.validation, self.test, self.column_indices)

    def on_validation_and_test(self, f):
        return Data(self.train, f(self.validation), f(self.test), self.column_indices)

    def normalize(self):
        mean = self.train.mean()
        std = self.train.std()
        return self.map(lambda df: (df - mean) / std)
