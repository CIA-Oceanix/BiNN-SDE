# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 08:44:49 2020

@author: noura
"""
#to delete all past variable

import numpy as np
import sdeint
from scipy.integrate import odeint
from AnDA_dynamical_models import AnDA_Lorenz_63

    
def generate_LS(GD):
#np.random.seed(1),
    def Stoch_Lorenz_63(S,t,sigma=GD.sigma, rho = GD.rho, beta = GD.beta, gama = GD.gama):
        x_1 = sigma*(S[1]-S[0])-4*S[0]/(2*gama)
        x_2 = S[0]*(rho-S[2])-S[1] -4*S[1]/(2*gama)
        x_3 = S[0]*S[1] - beta*S[2] -8*S[2]/(2*gama)
        dS  = np.array([x_1,x_2,x_3])
        return dS
#   ----------------------------------------------------------------------------------------- 
    def brownian_process(S,t,sigma=GD.sigma, rho = GD.rho, beta = GD.beta, gama = GD.gama):
        x_1 = 0.0
        x_2 = (rho - S[2])/np.sqrt(gama)
        x_3 = (S[1])/np.sqrt(gama)
        G = np.zeros((3,3))
        G[0,0] = x_1
        G[1,1] = x_2
        G[2,2] = x_3
        #np.fill_diagonal(G,dS),
        return G
    
    x0 = np.array([8.0,0.0,30.0])
    S = odeint(AnDA_Lorenz_63,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.sigma,GD.rho,GD.beta));
    x0 = S[S.shape[0]-1,:];

    tspan = np.arange(0,GD.N+0.000001,GD.dt_integration)
    S = sdeint.itoEuler(Stoch_Lorenz_63, brownian_process, x0, tspan)
    return S
