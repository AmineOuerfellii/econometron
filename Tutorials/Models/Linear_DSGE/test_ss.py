

import pandas as pd
import numpy as np
from econometron.Models.dynamicsge import linear_dsge
from econometron.Models.StateSpace import SS_Model
from econometron.utils.data_preparation import TransformTS
import os
from econometron.filters import Kalman,kalman_objective

data_dir= os.path.join(os.path.dirname(__file__), "../../Data_Fred/")

GDP = pd.read_csv(os.path.join(data_dir, "gdp.csv"), index_col=0, parse_dates=True)

# Model setup
equations=[ 
           "- r_t + phi*p_t=0",
           "p_t - beta * p_tp1 - kappa * (x_t - xbar_t) = 0",
           "x_t - x_tp1 + (1/g) * (r_t - p_tp1) = 0",
           "- xbar_tp1 + rho * xbar_t + sigmax = 0"]
variables=[ 'r','p','x','xbar']
states=['xbar']
exo_states=['xbar']
shock=['sigmax']
# Parameters dictionary
parameters = {
    'g': 5,      # Inverse of relative risk aversion (1/g)
    'beta': 0.99,       # Discount factor
    'kappa': 0,
    'rho': 0.95,        # Persistence of output gap target
    'phi': 1.5,         # Taylor rule inflation coefficient
    'd': 0.5,          # Calvo parameter
    'sigmax':0.01
}

sigma_X,beta,g,rho,phi,d=parameters['sigmax'],parameters['beta'],parameters['g'],parameters['rho'],parameters['phi'],parameters['d']
parameters['kappa']=((1-d)*(1-d*beta))/d
new_keynisian_model=linear_dsge(equations=equations,variables=variables,exo_states=exo_states,shocks=shock,parameters=parameters)
initial_guess = [1, 1, 1]
new_keynisian_model.set_initial_guess(initial_guess)
new_keynisian_model.compute_ss(method='fsolve',options={'xtol': 1e-10})
A,B,C=new_keynisian_model.approximate(method='analytical')
new_keynisian_model.solve_RE_model()
print("Policy Function (f):\n", new_keynisian_model.f)
print("State Transition (p):\n", new_keynisian_model.p)

#################"

GDP = pd.read_csv(os.path.join(data_dir, "gdp.csv"), index_col=0, parse_dates=True)
Inflation=pd.read_csv(os.path.join(data_dir, "inflation.csv"), index_col=0, parse_dates=True)
Interest_Rate=pd.read_csv(os.path.join(data_dir, "int_rate.csv"), index_col=0, parse_dates=True)
mac_data = pd.concat([Interest_Rate, Inflation, GDP], axis=1)
mac_data = mac_data.dropna()
# Rename columns for clarity
mac_data.columns = ['Int', 'Inf', 'GDP']
# Apply log transformation01)
mac_data = np.log(mac_data)
# Remove rows with -inf, inf, or NaN values
mac_data = mac_data.replace([np.inf, -np.inf], np.nan).dropna().transpose()
#int for intrest_rate for simplcity"
TSP=TransformTS(mac_data.T,method='hp',lamb=1600)
TSP.get_transformed_data().describe()
trans_data=TSP.get_transformed_data().T.values
model_full_params = {
    'g': 1.00000000e+01,
    'beta': 8.97384125e-01,
    'kappa': 0.8,
    'rho': 9.61923424e-01,
    'phi': 1,
    'd': 8.64607398e-01 ,
    'sigmax': 7.52359617e-03,
    'sigma_y': 0.01,
    'sigma_p': 0.01,
    'sigma_r': 0.01
}

Keynes_ss=SS_Model(data=trans_data,parameters=model_full_params,model=new_keynisian_model,optimizer='SA')
def R_mat(p):
    A1 = np.array([[1, -p ['phi'], 0],
                    [0, 1, -p['kappa']],
                    [1/p ['g'], 0, 1]])

    R1= np.array([[p['sigma_r'], 0, 0],
                [0, p['sigma_p'], 0],
                [0, 0, p['sigma_y']]])
    R=np.linalg.solve(A1,R1)
    return R
def Q_mat(p):
    Q=np.array([[p['sigmax']]])
    return Q
defined_params=({'kappa':'((1 - d) * (1 - d * beta)) / d'})
calibrated_params=[('beta',0.96)]


##########"
Keynes_ss.set_state_cov(Q_mat)
Keynes_ss.set_obs_cov(R_mat)
Keynes_ss.calibrate_params(calibrated_params)
Keynes_ss.define_parameter(defined_params)
update_ss=Keynes_ss._make_state_space_updater(base_params=model_full_params)
param={'g': 7.372995153121833,
 'beta': 0.96,
 'kappa': 0.8,
 'rho': 0.9877384102847888,
 'phi': 3.402829610145752,
 'd': 0.003082995908878812,
 'sigmax': 0.006405250449062197,
 'sigma_y': 0.10709796645551589,
 'sigma_p': 0.004041130660498564,
 'sigma_r': 0.999999142653202}
print(param['sigmax']**2)
update_ss=Keynes_ss._make_state_space_updater(base_params=param)
print(update_ss(params=param))

ll=kalman_objective(param.values(),{},param.keys(),trans_data,update_ss)
# LB= [0,0,1,0,0,0,0,0]
# UB= [10,1,5,1,1,1,1,1]
# Keynes_ss.fit(Lower_bound=LB,Upper_bound=UB,)
# print(Keynes_ss.result)
# Keynes_ss.summary()

