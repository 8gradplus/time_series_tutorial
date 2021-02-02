from pandas import DataFrame
from time_series.entities import Data


def split_data(df: DataFrame):
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    return Data(train=df[0:int(n*0.7)],
                validation=df[int(n*0.7):int(n*0.9)],
                test=df[int(n*0.9):],
                column_indices=column_indices)
