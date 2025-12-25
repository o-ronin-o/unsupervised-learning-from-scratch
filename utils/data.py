import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

class BreastCancerLoader:
    
    def __init__(self):
        self._data = load_breast_cancer()
        self.feature_names = self._data.feature_names
        self.target_names = self._data.target_names
        self.X = self._data.data
        self.y = self._data.target
        self.scaler = StandardScaler()

    def get_raw_data(self):
        
        return self.X, self.y

    def get_scaled_data(self):
        X_scaled = self.scaler.fit_transform(self.X)
        return X_scaled, self.y

    def get_dataframe(self):
        
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df['target'] = self.y
        return df

    def get_dims(self):
        return self.X.shape