import os

from google.cloud import storage
from termcolor import colored
from pasture_predict.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION


def storage_upload(filename, filetype = "joblib", rm=False):
    client = storage.Client().bucket(BUCKET_NAME)

    if filetype == "joblib":
        local_model_name11 = f"{filename}model11.joblib"
        storage_location11 = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name11}"
        blob = client.blob(storage_location11)
        blob.upload_from_filename(f"{filename}model11.joblib")
        print(colored(f"=> model.joblibs uploaded to bucket {BUCKET_NAME} inside {storage_location11}",
                    "green"))

        local_model_name1 = f"{filename}model1.joblib"
        storage_location1 = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name1}"
        blob = client.blob(storage_location1)
        #blob = client.blob(storage_location1)
        blob.upload_from_filename(f"{filename}model1.joblib")
        print(colored(f"=> model.joblibs uploaded to bucket {BUCKET_NAME} inside {storage_location1}",
                    "green"))

    if filetype != "joblib":
        storage_location = f"data/predict_{filename}_1.csv"
        blob = client.blob(storage_location)
        blob.upload_from_filename(f"raw_data/prect_{filename}_1.csv")
        print(colored(f"=> data {filename} uploaded to bucket {BUCKET_NAME} inside {storage_location}",
                    "blue"))

        storage_location = f"data/predict_{filename}_11.csv"
        blob = client.blob(storage_location)
        blob.upload_from_filename(f"raw_data/prect_{filename}_11.csv")
        print(colored(f"=> data {filename} uploaded to bucket {BUCKET_NAME} inside {storage_location}",
                    "blue"))

    if rm:
        if filetype == "joblib":
            os.remove(f"{filename}model11.joblib")
            os.remove(f"{filename}model1.joblib")

        if filetype != "joblib":
            os.remove(f"raw_data/prect_{filename}_1.csv")
            os.remove(f"raw_data/prect_{filename}_11.csv")
