import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import copy

class generic_basis(nn.Module):
    def __init__(self,backcast_length,forecast_length):
      super(generic_basis, self).__init__()
      self.backcast_length = backcast_length
      self.forecast_length = forecast_length
      ###
      #y_Hat_{l}=V_{l}**f * theta_{l}**f
      #x_Hat_{l}=V_{l}**b * theta_{l}**b
      self.basis_f = nn.Parameter(torch.randn(forecast_length,forecast_length) * 0.01)
      self.basis_b= nn.Parameter(torch.randn(backcast_length,backcast_length) * 0.01)
      # Bias terms
      self.b_f = nn.Parameter(torch.zeros(forecast_length))
      self.b_b = nn.Parameter(torch.zeros(backcast_length))
      ##
    def forward(self,theta_b,theta_f):
      print("theta_b shape:", theta_b.shape)
      print("theta_f shape:", theta_f.shape)
      print("basis_f shape:", self.basis_f.shape)
      print("basis_b shape:", self.basis_b.shape)
      backcast=torch.matmul(theta_b,self.basis_b)+self.b_b
      forecast=torch.matmul(theta_f,self.basis_f)+self.b_f
      return backcast,forecast
class polynomial_basis(nn.Module):
    def __init__(self, degree, backcast_length, forecast_length):
        super(polynomial_basis, self).__init__()
        self.degree = degree
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        #####
        #let's define the basis
        # we begin mohaaaa making normalization
        #so the vector of poly is the time index
        T_forecast_prime = np.linspace(0, 1, forecast_length + 1)[:-1]
        T_backcast_prime = np.linspace(0, 1, backcast_length + 1)[:-1]
        basis_forecast = np.zeros((degree + 1, forecast_length))  # Changed: (degree+1, forecast_length)
        basis_backcast = np.zeros((degree + 1, backcast_length))  # Changed: (degree+1, backcast_length)
        for i in range(degree + 1):
            basis_forecast[i, :] = T_forecast_prime ** i
            basis_backcast[i, :] = T_backcast_prime ** i
        self.register_buffer('forecast_basis', torch.tensor(basis_forecast, dtype=torch.float32))
        self.register_buffer('backcast_basis', torch.tensor(basis_backcast, dtype=torch.float32))

    def forward(self, theta_b, theta_f):
        print("theta_b shape:", theta_b.shape)
        print("theta_f shape:", theta_f.shape)
        print("basis_f shape:", self.forecast_basis.shape)
        print("basis_b shape:", self.backcast_basis.shape)
        forecast = torch.matmul(theta_f, self.forecast_basis)
        backcast = torch.matmul(theta_b, self.backcast_basis)
        return forecast, backcast
class chebyshev_basis(nn.Module):
      """
      this is experimental thing , I "the author of this package",
      since for interprebality we use a polyniomial basis , i thought we could use the cheb basis , since it more
      performant.
      """
      def __init__(self,
                   backcast_length,
                   forecast_length,
                   degree):
        super(chebyshev_basis, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.degree = degree
        t_back = np.linspace(-1, 1, backcast_length)
        t_fore = np.linspace(-1, 1, forecast_length)
        basis_back = np.zeros((degree + 1,backcast_length))
        basis_fore = np.zeros((degree + 1,forecast_length))
        for i in range(backcast_length):
            basis_back[0,i] = 1.0
            if degree >= 1:
                basis_back[1,i] = t_back[i]
            for n in range(2, degree + 1):
                basis_back[n,i] = 2 * t_back[i] * basis_back[n-1,i] - basis_back[n-2,i]
        for i in range(forecast_length):
            basis_fore[0,i] = 1.0
            if degree >= 1:
                basis_fore[1,i] = t_fore[i]
            for n in range(2, degree + 1):
                basis_fore[n,i] = 2 * t_fore[i] * basis_fore[n-1,i] - basis_fore[n-2,i]
        self.register_buffer('forecast_basis', torch.tensor(basis_fore, dtype=torch.float32))
        self.register_buffer('backcast_basis', torch.tensor(basis_back, dtype=torch.float32))
      def forward(self, theta_b,theta_f):
          forecast = torch.matmul(theta_b,self.forecast_basis)
          backcast = torch.matmul(theta_f,self.backcast_basis)
          return backcast, forecast
class fourier_basis(nn.Module):
    def __init__(self, backcast_length, forecast_length):
        super(fourier_basis, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        # match the reference in the Nbeats paper
        #inchallah in the next update ,i'll give the user the choice for harmonics but now leave that way
        self.H_back = backcast_length // 2 - 1
        self.H_fore = forecast_length // 2 - 1     
        # print("H_back:", self.H_back)
        # print("H_fore:", self.H_fore)
        self.basis_size_back = 2 * self.H_back
        self.basis_size_fore = 2 * self.H_fore
        t_back = np.arange(backcast_length, dtype=np.float32) / backcast_length
        t_fore = np.arange(forecast_length, dtype=np.float32) / forecast_length
        basis_back = np.zeros((self.basis_size_back, backcast_length))
        basis_fore = np.zeros((self.basis_size_fore, forecast_length))
        for l in range(1, self.H_back + 1):
            basis_back[2*(l-1), :] = np.cos(2 * np.pi * l * t_back)
            basis_back[2*(l-1)+1, :] = np.sin(2 * np.pi * l * t_back)
        for l in range(1, self.H_fore + 1):
            basis_fore[2*(l-1), :] = np.cos(2 * np.pi * l * t_fore) 
            basis_fore[2*(l-1)+1, :] = np.sin(2 * np.pi * l * t_fore)    
        self.register_buffer('backcast_basis', torch.FloatTensor(basis_back))
        self.register_buffer('forecast_basis', torch.FloatTensor(basis_fore))
    
    def forward(self, theta_b, theta_f):
        forecast = torch.matmul(theta_f, self.forecast_basis)
        backcast = torch.matmul(theta_b, self.backcast_basis)
        return forecast, backcast
    
class N_beats_Block(nn.Module):
    def __init__(self,input_size,Horizon,backcast,degree,
                    Hidden_size=512,
                    basis_type="generic"):
        super(N_beats_Block,self).__init__()
        self.basis_type=basis_type
        self.input_size=input_size
        self.degree=degree
        self.Horizon=Horizon
        self.backcast=backcast
        #####
        # we will set 4 layers for teh Fully connected stack 
        self.FC_stack=nn.Sequential(
            nn.Linear(in_features=input_size,out_features=Hidden_size),
            nn.ReLU(),#h(l1)
            nn.Linear(in_features=Hidden_size,out_features=Hidden_size),
            nn.ReLU(),#h(l2)
            nn.Linear(in_features=Hidden_size,out_features=Hidden_size),
            nn.ReLU(),#h(l3)
            nn.Linear(in_features=Hidden_size,out_features=Hidden_size),
            nn.ReLU() #h(l4)  
            )
        #see page 3 in N-BEATS paper by Boris N. Oreshkin
        # Now we prepare for the FC layer within each basis type choice  
        #please contact me @mohamedamine.ouerfelli@outlook.com specially 
        #if the matter concerns the architecture of the model and the single FC layer in the Nbeats block with generates theta
        if basis_type in ['generic','fourier','chebyshev','polynomial']:
            self.basis=basis_type
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")
        if self.basis=='generic':
            self.theta_f=nn.Linear(in_features=Hidden_size,out_features=Horizon)
            self.theta_b=nn.Linear(in_features=Hidden_size,out_features=backcast)
            self.basis_function=generic_basis(backcast_length=backcast,forecast_length=Horizon)
        elif self.basis=='fourier':
            theta_Hor=self.Horizon//2 - 1
            theta_back=self.backcast//2 - 1
            self.theta_f=nn.Linear(in_features=Hidden_size,out_features=2*theta_Hor)
            self.theta_b=nn.Linear(in_features=Hidden_size,out_features=2*theta_back)
            self.basis_function=fourier_basis(backcast_length=backcast,forecast_length=Horizon)
        elif self.basis=='chebyshev':
            self.theta_f=nn.Linear(in_features=Hidden_size,out_features=self.degree+1)
            self.theta_b=nn.Linear(in_features=Hidden_size,out_features=self.degree+1)
            self.basis_function=chebyshev_basis(backcast_length=backcast,forecast_length=Horizon,degree=degree)
        elif self.basis=='polynomial':
            self.theta_f=nn.Linear(in_features=Hidden_size,out_features=self.degree+1)
            self.theta_b=nn.Linear(in_features=Hidden_size,out_features=self.degree+1)
            self.basis_function=polynomial_basis(backcast_length=backcast,forecast_length=Horizon,degree=degree)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self,x):
        h_4=self.FC_stack(x)
        theta_b=self.theta_b(h_4)
        theta_f=self.theta_f(h_4)
        forecast,backcast=self.basis_function(theta_b,theta_f)
        return forecast,backcast
