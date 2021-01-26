from matplotlib import pyplot as plt
import numpy as np


def plot_time_series(time, series, model_series=None, mse=None, y_label=None, x_lim=None, style="-"):
    fig, ax = plt.subplots(figsize=(15, 3))
    start = 0
    end = None
    ax.plot(time[start:end], series[start:end], style,  color="red",  alpha=.4, label="time series")
    ax.set_xlabel("Time")
    if not y_label:
        y_label = "Value"
    ax.set_ylabel(y_label)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.grid(True)
    if model_series is not None:
        label = "forecast"
        if mse:
            rounded_mse = np.round(mse, 2)
            label = label + " (mse={mse})".format(mse=rounded_mse)
        ax.plot(time[start:end], model_series[start:end], style,  color="black",  alpha=.4, label=label)
        ax.legend()
