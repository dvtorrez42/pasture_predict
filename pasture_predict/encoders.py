import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class interpolateEncoder():

#    def __init__(self):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.rad = X_.rad.interpolate()
        X_.temp_2m = X_.temp_2m.interpolate()
        X_.temp_sup = X_.temp_sup.interpolate()
        X_.hum_arriba = X_.hum_arriba.interpolate()
        X_.hum_raiz = X_.hum_raiz.interpolate()
        return X_[['rad','temp_2m','temp_sup','hum_arriba','hum_raiz']]

class maEncoder():

#    def __init__(self):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = X.index(X)
        X_.rad = X_.rad.interpolate()
        X_.temp_2m = X_.temp_2m.interpolate()
        X_.temp_sup = X_.temp_sup.interpolate()
        X_.hum_arriba = X_.hum_arriba.interpolate()
        X_.hum_raiz = X_.hum_raiz.interpolate()
        return X_[['rad','temp_2m','temp_sup','hum_arriba','hum_raiz']]
