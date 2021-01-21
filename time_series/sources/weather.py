import pandas as pd
import os
import tensorflow as tf


def get_weather_data():
    """ Retrieve weather data.
    From tensor flow tutorial https://www.tensorflow.org/tutorials/structured_data/time_series?hl=en
    """
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        cache_dir="~/",
        cache_subdir="resources",
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    # subsample 10 min -> 1 hour intervals
    df = df[5::6]
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    df.index = date_time
    return df
