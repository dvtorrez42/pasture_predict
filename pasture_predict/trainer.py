import joblib
from termcolor import colored
import mlflow
from pasture_predict.data import get_data, prepare_data
from pasture_predict.encoders import interpolateEncoder
from pasture_predict.gcp import storage_upload
from pasture_predict.utils import compute_mape
from pasture_predict.params import MLFLOW_URI, EXPERIMENT_NAME, BUCKET_NAME, MODEL_VERSION, MODEL_VERSION
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Trainer(object):
    def __init__(self,batch_name,  X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.bach = batch_name
        self.X = X
        self.y = y
        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # interpolatepipe = Pipeline([
        #     ('interpolate', interpolateEncoder())
        # ])
        # time_pipe = Pipeline([
        #     ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        #     ('ohe', OneHotEncoder(handle_unknown='ignore'))
        # ])
        # preproc_pipe = ColumnTransformer([
        #     ('distance', dist_pipe, [
        #         "pickup_latitude",
        #         "pickup_longitude",
        #         'dropoff_latitude',
        #         'dropoff_longitude'
        #     ]),
        #     ('time', time_pipe, ['pickup_datetime'])
        # ], remainder="drop")
        self.pipeline = Pipeline([
        #     ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

    def run(self):
        self.set_pipeline()
        self.mlflow_log_param("model", "LinearRegression")
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        mape = compute_mape(y_pred, y_test)
        self.mlflow_log_metric("mape", mape)
        return round(mape, 2)

    def save_model_locally(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, f"{self.batch_name}model.joblib")
        print(colored("model.joblib saved locally", "green"))

    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":
    # Get and clean data
    N = 10000
    batch_name = "vieytes"
    df = get_data(batch_name,nrows=N)
    df = prepare_data(df)
    y = df["prod"]
    X = df.drop(columns=['prod','date','ENSOType'])
    X_train = X[:-22]
    y_train = y[:-22]
    X_test = X[-22:]
    y_test = X[-22:]

    # # Train and save model, locally and
    trainer = Trainer(batch_name,X=X_train, y=y_train)
    trainer.set_experiment_name('xp2')
    trainer.run()
    mape = trainer.evaluate(X_test, y_test)
    print(f"mape: {mape}")
    trainer.save_model_locally()
    storage_upload()
