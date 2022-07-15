### MLFLOW configuration - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[AR][BSAS][DAB] pasture_predict v1"

### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

PATH_TO_LOCAL_MODEL = 'model.joblib'

AWS_BUCKET_PATH = "s3://wagon-public-datasets/" #taxi-fare-test.csv"

DATASET_BATCH_SAN_LUIS = "dataset_completo_sanluis.csv"

DATASET_BATCH_VIEYTES = "dataset_completo_vieytes.csv"

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-840-pasture_predict'
#BUCKET_NAME = 'wagon-data-840-valente'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_DATA_PATH = 'data'
#BUCKET_DATA_PATH = 'pasture_predict'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'pasture_predict'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -

API_URL = 'http://127.0.0.1:8000/predict'

SITE_URL= 'http://172.28.230.58:8501/'
