import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from econometron.utils.data_preparation import TransformTS
from econometron.Models.VectorAutoReg.VAR import VAR

import os
print(os.path.exists("../../Data_Fred/gdp.csv"))
GDP = pd.read_csv("../../Data_Fred/gdp.csv", index_col=0, parse_dates=True)
Inflation = pd.read_csv("../../Data_Fred/inflation.csv", index_col=0, parse_dates=True)
Interest_Rate = pd.read_csv("../../Data_Fred/int_rate.csv", index_col=0, parse_dates=True)
mac_data=pd.concat([GDP, Inflation, Interest_Rate], axis=1).dropna()
TSP=TransformTS(mac_data,method='diff',analysis=True,plot=True)
data=TSP.get_transformed_data()
vm=VAR(data,max_p=10,check_stationarity=True,plot=True)
model_1=vm.fit(output=False)
####
vm.predict(10)