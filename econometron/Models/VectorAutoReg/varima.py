import numpy as np
import logging
import econometron.Models.VectorAutoReg.VARMA as VARMA
import econometron.utils.data_preparation.process_timeseries as TimeSeriesProcessor

class VARIMA(VARMA):
    
  def __init__(self,data,max_p=5,max_q=5,columns=None,forecast_h=6,plot=True,check_stationarity=True,bootstrap_n=1000,criterion='AIC',ci_alpha=0.05,Key=None,Threshold=0.8,orth=False):
    #making data stationnary :
    TSP=TimeSeriesProcessor(data,method='diff')
    self.data=TSP.get_transformed_data()
    self.stationarity_info=TSP.get_stationarity_info()
    self.I=max(info['order'] for info in self.stationarity_info.values())
    super().__init__(self.data,max_p,max_q,columns,forecast_h,plot,check_stationarity,bootstrap_n,criterion,False,ci_alpha,Key,Threshold,orth)
    self.best_model=None
  def fit(self,p=None,q=None,output=None):
    best_model=super().fit(p,q,output=False)
    



    




