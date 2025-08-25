from econometron.filters import Kalman
from typing import Union
import pandas as pd
import numpy as np
from econometron.Models.dynamicsge import linear_dsge
class SS_Model:
    def __init__(self,data:Union[np.ndarray, pd.DataFrame,pd.Series],parameters:dict, A:np.ndarray, C:np.ndarray, Q:np.ndarray, R:np.ndarray, P:np.ndarray, x0:np.ndarray,model:linear_dsge=None,optimizer:str='L-BFGS-B',estimation_method:str='MLE',constraints:dict=None):
        """
        Initializes the State Space Model with the given parameters.
        Parameters:
        - data (Union[np.ndarray, pd.DataFrame, pd.Series]): The observed data.
        - parameters (dict): Model parameters.
        - A (np.ndarray): State transition matrix.
        - C (np.ndarray): Observation matrix.
        - Q (np.ndarray): Process noise covariance.
        - R (np.ndarray): Measurement noise covariance.
        - P (np.ndarray): Estimate error covariance.
        - x0 (np.ndarray): Initial state estimate.
        - model (econometron.Models.dynamicsge.linear_dsge, optional): The linear DSGE model.
        - optimizer (str, optional): The optimization algorithm to use.
        - estimation_method (str, optional): The estimation method to use.
        - constraints (dict, optional): Constraints for the optimization.
        """
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            self.data = data.values
        else:
            self.data = data
        self.parameters = parameters 
        self.P = P
        self.x0 = x0
        self.model = model
        self.optimizer = optimizer
        self.technique = estimation_method
        self.constraints = constraints
    def set_transition_mat(self,A:Union[np.ndarray, np.matrix]):
        if A.shape[0] != A.shape[1]:
            raise ValueError("Transition matrix A must be square.")
        self.A = A
    def set_design_mat(self,C:Union[np.ndarray, np.matrix]):
        if self.A is None:
            raise ValueError("Set transition matrix  before setting observation matrix .")
        if C.shape[1] != self.A.shape[0] or C.shape[0] != self.data.shape[1]:
            raise ValueError("Observation matrix C dimensions are not compatible with transition matrix  or data.")
        self.C = C
    def set_obs_cov(self,Q:Union[np.ndarray, np.matrix]):
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Observation covariance matrix Q must be square.")
        self.Q = Q
    def set_state_cov(self,R:Union[np.ndarray, np.matrix]):
        if R.shape[0] != R.shape[1]:
            raise ValueError("State covariance matrix R must be square.")
        self.R = R
    def _validate_entries_(self):
        if self.technique=='Bayesian':
            if self.optimizer is not None:
                self.optimizer = None
        if self.constraints and self.optimizer != 'trust-constr' :
            print("unable to account for constraints")
        if self.technique == 'MLE':
            if self.optimizer is None:
                self.optimizer = 'SA'
        

    def fit(self):
        # Fit the model to the data
        pass
    
    def summary(self):
        # Generate a summary of the model's parameters and state
        pass
    
    def predict(self, steps: int):
        # Generate predictions for the next 'steps' time points
        pass