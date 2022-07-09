import joblib
from termcolor import colored
import mlflow
from pasture_predict.data import get_data, save_data, prepare_data
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
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor

class Trainer(object):
    def __init__(self,batch_name, df_train,X,y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        #self.pipeline = None
        self.model = None
        self.batch_name = batch_name
        self.df_train = df_train
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
            ('model', RandomForestRegressor(n_estimators=40, random_state=0))
        ])

    def set_model(self):
        self.model = Prophet(seasonality_mode='multiplicative', interval_width=0.95)

    def run(self):
        self.set_pipeline()
        self.set_model()
        self.mlflow_log_param("model", "Prophet")
        self.model.fit(self.df_train)
        self.pipeline.fit(self.X,self.y)

    def evaluate(self, y11_test, X1_test,y1_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        mape = []
        future = self.model.make_future_dataframe(periods=11, freq='8D')
        forecast = self.model.predict(future)
        self.y11_pred = forecast.yhat[-11:].values
        mape.append(round(mean_absolute_percentage_error(y11_test, self.y11_pred),2))

        self.y1_pred = self.pipeline.predict(X1_test)
        mape.append(round(mean_absolute_percentage_error(y1_test, self.y1_pred),2))

        #self.mlflow_log_metric("mape", mape)
        return mape

    def save_model_locally(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.model, f"{self.batch_name}model11.joblib")
        joblib.dump(self.pipeline, f"{self.batch_name}model1.joblib")
        print(colored("model.joblib saved locally", "green"))

    def save_model_csv(self):
        """Save the model into a .csv"""
        save_data( self.y1_pred, self.y11_pred , batch_name)
        #joblib.dump(self.model, f"{self.batch_name}model11.joblib")
        #joblib.dump(self.pipeline, f"{self.batch_name}model1.joblib")
        print(colored("data saved locally", "blue"))

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
    batch_name = "vieytes"
    df = get_data(batch_name)
    df = prepare_data(df)

    df11 = df.copy()
    df11 = df11.rename(columns={'date':'ds', 'prod':'y'})
    df11_train = df11[:-11]
    y11_test = df11[-11:]['y']

    df1 = df.copy()
    df1_train = df[:-1]
    df1_test = df[-1:]
    X1_train = df1_train.drop(['date', 'prod','ENSOType'],axis=1)
    y1_train = df1_train['prod']
    X1_test = df1_test.drop(['date', 'prod','ENSOType'],axis=1)
    y1_test = df1_test['prod']

    # # Train and save model, locally and
    trainer = Trainer(batch_name=batch_name,df_train=df11_train,X=X1_train,y=y1_train)
    trainer.set_experiment_name('xp2')
    trainer.run()
    mape = trainer.evaluate(y11_test,X1_test,y1_test)
    print(f"mape 11 valores: {mape[0]}")
    print(f"mape 1 valor acum: {mape[1]}")
    trainer.save_model_locally()
    trainer.save_model_csv()
    storage_upload(filename=batch_name,filetype="joblib")
    storage_upload(filename=batch_name,filetype="csv")

    df = get_data(batch_name, f"predict_11")
