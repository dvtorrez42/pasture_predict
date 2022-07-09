
# $DELETE_BEGIN
from datetime import datetime
import pytz

import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pasture_predict.data import get_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2


@app.get("/")
def index():
    return dict(greeting="hello")


@app.get("/predict")
def predict(batch_name, predict=1):

    # create datetime object from user provided date
    ##pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # localize the user provided datetime with the NYC timezone
    ##eastern = pytz.timezone("US/Eastern")
    ##localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # convert the user datetime to UTC
    ##utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    # format the datetime as expected by the pipeline
    ##formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    # fixing a value for the key, unused by the model
    # in the future the key might be removed from the pipeline input
    # eventhough it is used as a parameter for the Kaggle submission
    ##key = "2013-07-06 17:18:00.000000119"

    # build X ⚠️ beware to the order of the parameters ⚠️
    # X = pd.DataFrame(dict(
    #     key=[key],
    #     pickup_datetime=[formatted_pickup_datetime],
    #     pickup_longitude=[float(pickup_longitude)],
    #     pickup_latitude=[float(pickup_latitude)],
    #     dropoff_longitude=[float(dropoff_longitude)],
    #     dropoff_latitude=[float(dropoff_latitude)],
    #     passenger_count=[int(passenger_count)]))

    # ⚠️ TODO: get model from GCP

    # pipeline = get_model_from_gcp()
    #model = joblib.load(f"{batch_name}model11.joblib")
    #pipeline = joblib.load(f"{batch_name}model1.joblib")

    # make prediction
    #results11 = model.predict(batch_name)
    #results1 = pipeline.predict(batch_name)

    # convert response from numpy to python type
    #pred11 = float(results11)
    #pred1 = float(results1)

    df = get_data(batch_name, f"predict_{predict}")
    return(df)

    #return batch_name
    #return True
# $DELETE_END
