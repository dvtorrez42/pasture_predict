import pandas as pd
import numpy as np

from pasture_predict.utils import simple_time_tracker
from google.cloud import storage
from pasture_predict.params import BUCKET_NAME, BUCKET_DATA_PATH


@simple_time_tracker
def get_data(batch_name="vieytes", type = "data", nrows=10000, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    if type == "data":
        dataset_name = f"dataset_completo_{batch_name}.csv"

    if type == "predict_1":
        dataset_name = f"predict_{batch_name}_1.csv"

    if type == "predict_11":
        dataset_name = f"predict_{batch_name}_11.csv"

    if type == "predict_200":
        dataset_name = f"predict_{batch_name}_200.csv"

    #dataset_name ="predict_sanluis_200.csv"

    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}/{dataset_name}"
    #return path
    df = pd.read_csv(path)# , nrows=nrows)
    return df

def save_data( dict1 = [], dict11 = [] , filename="vieytes"):
    """method to save de predict dict to csv"""
    if len(dict1)>=1:
        dataset_name = f"prect_{filename}_1"
        df = pd.DataFrame.from_dict(dict1)
        df.columns =["data"]
        df.to_csv (f'raw_data/{dataset_name}.csv', index = False, header=True)

    if len(dict11)>=1:
        dataset_name = f"prect_{filename}_11"
        df = pd.DataFrame.from_dict(dict11)
        df.columns =["data"]
        df.to_csv (f'raw_data/{dataset_name}.csv', index = False, header=True)

def clean_data(df):
    df.loc[df.rad < 0,'rad'] = np.nan
    df.loc[(df.temp_2m < -50) | (df.temp_2m < -50),'temp_2m'] = np.nan
    df.loc[(df.temp_sup < -50) | (df.temp_sup < -50),'temp_sup'] = np.nan
    df.loc[df.hum_arriba < 0,'hum_arriba'] = np.nan
    df.loc[df.hum_raiz < 0,'hum_raiz'] = np.nan

    df.rad = df.rad.interpolate()
    df.temp_2m = df.temp_2m.interpolate()
    df.temp_sup = df.temp_sup.interpolate()
    df.hum_arriba = df.hum_arriba.interpolate()
    df.hum_raiz = df.hum_raiz.interpolate()

    df.date =  pd.to_datetime(df.date)
    df = df[df.date < '2022-02-20']
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

def make_acum(df):
    df2 = df.copy()

    df2['prod_acumulada'] = df2['prod'].rolling(window=11,closed='left').sum().shift(-11)
    df2 = df2.dropna()
    return df2

def prepare_data(df):
    df = clean_data(df)
    df = make_ml_approach(df)
    df = make_ma(df)
    df = make_ewma(df)
    df = make_acum(df)
    return df

if __name__ == '__main__':
    df = get_data()
    df = prepare_data(df)
    print("done")
