import numpy as np
from matplotlib import pyplot as plt

from time_series.entities import Offsets
from time_series.helpers.dataset import extract_labels, extract_features, to_array


class PlotSingleTimeSeries:
    def __init__(self, offsets: Offsets, input_column_index: int,   model=None):
        self.offsets = offsets
        self.input_column_index = input_column_index
        self.model = model

    def __call__(self, iterator):
        data = iterator.get_next()
        x_input = data[0]
        y_true = data[1]
        y_predicted = None
        if self.model:
            y_predicted = self.model.predict(x_input).flatten()
        self.plot(x_input.numpy()[:, :, self.input_column_index].flatten(), y_true.numpy().flatten(), y_predicted)

    def from_dataset(self, ds):
        """Plot single element ds"""
        x_input = to_array(extract_features(ds))
        x_input = x_input[:, self.input_column_index].flatten()
        y_true = to_array(extract_labels(ds))
        y_true = y_true.flatten()
        y_predicted = None
        if self.model:
            y_predicted = self.model.predict(ds).flatten()
        self.plot(x_input, y_true, y_predicted)

    def plot(self, x_input, y_true, y_predicted):
        """Plots single time series as specified by offsets"""

        times_features = np.arange(self.offsets.input_width)
        start = self.offsets.input_width + self.offsets.shift - self.offsets.label_width
        end = start + self.offsets.label_width
        times_forecast = np.arange(start, end)
        fix, ax = plt.subplots(figsize=(15, 3))
        ax.plot(times_features, x_input, "-o", color="black", ms=10, mfc="white", label="Input")
        ax.plot(times_forecast, y_true, "-o", color="black", ms=6, label="Truth")
        if self.model:
            ax.plot(times_forecast, y_predicted, "-o", color="red", ms=6,  label="Forecast")
        ax.legend()
        ax.set_xticks(np.arange(0, end))
