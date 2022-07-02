import pandas as pd
import numpy as np

from pasture_predict.utils import simple_time_tracker
from google.cloud import storage
from pasture_predict.params import BUCKET_NAME, BUCKET_DATA_PATH


@simple_time_tracker
def get_data(batch_name="vieytes", nrows=10000, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    dataset_name = f"dataset_completo_{batch_name}.csv"

    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}/{dataset_name}"
    df = pd.read_csv(path, nrows=nrows)
    return df

def clean_data(df):
    df.rad[df.rad < 0] = np.nan
    df.temp_2m[(df.temp_2m < -50) | (df.temp_2m < -50)] = np.nan
    df.temp_sup[(df.temp_sup < -50) | (df.temp_sup < -50)] = np.nan
    df.hum_arriba[df.hum_arriba < 0] = np.nan
    df.hum_raiz[df.hum_raiz < 0] = np.nan

    df.rad = df.rad.interpolate()
    df.temp_2m = df.temp_2m.interpolate()
    df.temp_sup = df.temp_sup.interpolate()
    df.hum_arriba = df.hum_arriba.interpolate()
    df.hum_raiz = df.hum_raiz.interpolate()
    return df

def make_ma(df):
    df['ma_6'] = df['prod'].rolling(window=6).mean()
    df['ma_12'] = df['prod'].rolling(window=12).mean()
    df['ma_24'] = df['prod'].rolling(window=24).mean()
    df['ma_32'] = df['prod'].rolling(window=32).mean()
    df['ma_46'] = df['prod'].rolling(window=46).mean()
    return df

def make_ewma(df):
    df["ewma_6"] = df["prod"].ewm(halflife=6).mean()
    df["ewma_12"] = df["prod"].ewm(halflife=12).mean()
    df["ewma_24"] = df["prod"].ewm(halflife=24).mean()
    df["ewma_32"] = df["prod"].ewm(halflife=32).mean()
    df["ewma_46"] = df["prod"].ewm(halflife=46).mean()
    return df

def make_ml_approach (df):
    df2 = df.copy()

    for i in range(1, 47):
        df2[f'x_{i}'] = df2["prod"].shift(i)

    # Drop nan
    df2 = df2.dropna()
    return df2

def prepare_data(df):
    df = clean_data(df)
    df = make_ml_approach(df)
    df = make_ma(df)
    df = make_ewma(df)
    return df

if __name__ == '__main__':
    df = get_data()
    df = prepare_data(df)
    print("done")
