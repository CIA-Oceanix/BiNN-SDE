# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 08:44:49 2020

@author: noura
"""
#to delete all past variable

import numpy as np
import sdeint

    
def generate_GM(GD):
#np.random.seed(1),
    def OU_prcess(S,t,theta=GD.theta, mu = GD.mu, gama = GD.gama):
        dS = (mu*S)+theta
       
        return dS
#   ----------------------------------------------------------------------------------------- 
    def brownian_process(S, t,gama = GD.gama):
     
        return (gama*S)
    
    x0 =0.01
    # x0 =0.000001



    tspan = np.arange(0,GD.N+0.0000001,GD.dt_integration)
    S = sdeint.itoEuler(OU_prcess, brownian_process, x0, tspan)
      # reinitialize random generator number
    np.random.seed()

    return S
