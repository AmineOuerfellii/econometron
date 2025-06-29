import numpy as np
from scipy import stats

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None       
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)
        return self   
    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Must fit before transform")
        return (X - self.mean_) / self.std_ 
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    def inverse_transform(self, X):
        return X * self.std_ + self.mean_
class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.scale_ = self.max_ - self.min_
        self.scale_[self.scale_ == 0] = 1
        return self 
    def transform(self, X):
        if self.min_ is None or self.max_ is None:
            raise ValueError("Must fit before transform")
        X_scaled = (X - self.min_) / self.scale_
        min_range, max_range = self.feature_range
        return X_scaled * (max_range - min_range) + min_range
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    def inverse_transform(self, X):
        min_range, max_range = self.feature_range
        X_scaled = (X - min_range) / (max_range - min_range)
        return X_scaled * self.scale_ + self.min_
def standard_scale(X):    
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=0)

def minmax_scale(X, feature_range=(0, 1)):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1 
    X_scaled = (X - min_val) / range_val
    min_range, max_range = feature_range
    return X_scaled * (max_range - min_range) + min_range
def standard_scale_scipy(X):
    return stats.zscore(X, axis=0)
# ===== METRICS =====
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)