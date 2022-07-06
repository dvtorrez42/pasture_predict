import time
import numpy as np

def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def compute_mape(y_pred, y_true):
    return np.mean(np.abs(np.array(y_pred) - np.array(y_true))/np.abs(np.array(y_true)))

################
#  DECORATORS  #
################

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed
