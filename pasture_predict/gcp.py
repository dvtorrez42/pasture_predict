import os

from google.cloud import storage
from termcolor import colored
from pasture_predict.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION


def storage_upload(filename, rm=False):
    client = storage.Client().bucket(BUCKET_NAME)

    local_model_name11 = f"{filename}model11.joblib"
    storage_location11 = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name11}"
    blob = client.blob(storage_location11)
    blob.upload_from_filename(f"{filename}model11.joblib")
    print(colored(f"=> model.joblibs uploaded to bucket {BUCKET_NAME} inside {storage_location11}",
                  "green"))

    local_model_name1 = f"{filename}model1.joblib"
    storage_location1 = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name1}"
    blob = client.blob(storage_location1)
    blob = client.blob(storage_location1)
    blob.upload_from_filename(f"{filename}model1.joblib")
    print(colored(f"=> model.joblibs uploaded to bucket {BUCKET_NAME} inside {storage_location1}",
                  "green"))
    if rm:
        os.remove(f"{filename}model11.joblib")
        os.remove(f"{filename}model1.joblib")
