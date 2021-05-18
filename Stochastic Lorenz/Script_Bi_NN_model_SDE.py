

from IPython import get_ipython
get_ipython().magic('reset -sf')
from AnDA_stat_functions import AnDA_RMSE
import numpy as np
import matplotlib.pyplot as plt
from parse_args_dkf import params 
from generate_LS import generate_LS
import sympy as sp



# parameters
class D_parameters:
    # LS parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0/3
#        noise
    gama=10
    N=10
# integration step
    dt_integration = 0.001 # integration time
    




# number of simulations
Nb_simu=20
# Parameters of the deterministic component  for the BiNN-SDE(F)
F_sde_all=np.zeros((Nb_simu,27))
# Parameters of the deterministic component  for the BiNN(BiNN classic)
F_class_all=np.zeros((Nb_simu,27))
# Parameters of the stochastic component  for the BiNN-SDE(L)
L_W_sde_all=np.zeros((Nb_simu,3,3))
L_B_sde_all=np.zeros((Nb_simu,3))

# initialize the RMSE for F and L
RMSE_BiNN_sde_F=np.zeros(Nb_simu)
RMSE_BiNN_sde_L=np.zeros(Nb_simu)

RMSE_BiNN_class=np.zeros(Nb_simu)


for k in range(Nb_simu):
    print(k)
    #   Generate data using Euler Maruymer method
    State=generate_LS(D_parameters)
    State_init=State
    N=State.shape[0]
    
#    ---------------------------------------
#    ----------------------------------------

    trainx=State[0:N-1,:]
    trainync= State[1:N,:]
#-----------------------------------------------------------------
#neural net params 
    params['transition_layers']=1
    params['bi_linear_layers']=3
    params['batch_size']=32
    params['ntrain']=[10,2500]
 
    
#------------------------------------------------------------------------
    from BiNN_model_SDE import dynamic_model
    pred_model=dynamic_model(D_parameters,trainx, trainync,params,None)
    

###-------------------------------------------------------------------------------------

####Generate trajectory using the learned model
 
    State_train=np.zeros((N,3))
    x0=State_init[0,:]
    State_train[0,:]=x0

    for i in range(N-1):
    
        tmp=np.reshape(x0,(1,3))
        ynew=pred_model.predict(tmp)
        State_train[i+1,:]=ynew[0]
        x0=State_train[i+1,:]
##        
## ===========================

##    ------------------------------------------------------------------------------
##    calcul RMSE
    W1=pred_model.get_layer('tensor_0').get_weights()[0].T
    B1=pred_model.get_layer('tensor_0').get_weights()[1]

    gama=D_parameters.gama

    L_W_sde_all[k,:,:]=W1
    L_B_sde_all[k,:]=B1
   # True weight and bias for L
    Wtrue=np.array([[0, 0,0],[0,0,-1/np.sqrt(gama)],[0,1/np.sqrt(gama),0]])
    Btrue=np.array([0,D_parameters.rho/np.sqrt(gama),0])

    RMSE_BiNN_sde_L[k]=AnDA_RMSE(Wtrue,W1)

## =============================================================================
## Calculate F from the model weight
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x1x2, x1x3, x2x3 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1x2 x1x3 x2x3')
    states=np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
    
    from get_dynamical_model import get_dynamical_model
    residual=get_dynamical_model(pred_model,3,3,1)

    true_weights_wb_sde=np.array([0,0,0,-10-(2/gama),0,0,10,0,0,0,0,-1,28,0,0,-1-(2/gama),0,0,0,1,0,0,0,0,0,0,-(8./3)-(4/gama)])
    a_wb=np.zeros((9*3))
    a_wb[:9] = (sp.Poly(residual[0][0], (x1,x2,x3))).coeffs()[:9]
    a_wb[9:18] = (sp.Poly(residual[1][0], (x1,x2,x3))).coeffs()[:9]
    a_wb[18:27] = (sp.Poly(residual[2][0], (x1,x2,x3))).coeffs()[:9]
    
    F_sde_all[k,:]=a_wb
#calcul RMSE
    RMSE_BiNN_sde_F[k]=AnDA_RMSE(true_weights_wb_sde,a_wb)


#
####------------------------------------------------------------
### =============================================================================
### BiNN clasic using the Euler method for intergration
### =============================================================================
##
    from Bi_NN_model import dynamic_model_class
    pred_model_class=dynamic_model_class(D_parameters,trainx, trainync,params,None)

#### =============================================================================
# Generate trajectory using the learned model with BiNN 
    State_train_class=np.zeros((N,3))
    x0=State_init[0,:]
    State_train_class[0,:]=x0

    for i in range(N-1):
    
        tmp=np.reshape(x0,(1,3))
        ynew=pred_model_class.predict(tmp)
        State_train_class[i+1,:]=ynew[0]
        x0=State_train_class[i+1,:]
          

# =============================================================================
# Calculte F from the model weights
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x1x2, x1x3, x2x3 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1x2 x1x3 x2x3')
    states=np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
   
    residual=get_dynamical_model(pred_model_class,3,3,1)
####comparing equations without biaises 

    a_wb_class=np.zeros((9*3))
    a_wb_class[:9] = (sp.Poly(residual[0][0], (x1,x2,x3))).coeffs()[:9]
    a_wb_class[9:18] = (sp.Poly(residual[1][0], (x1,x2,x3))).coeffs()[:9]
    a_wb_class[18:27] = (sp.Poly(residual[2][0], (x1,x2,x3))).coeffs()[:9]
    F_class_all[k,:]=a_wb_class

    
#calcul RMSE
    RMSE_BiNN_class[k]=AnDA_RMSE(true_weights_wb_sde,a_wb_class)

# #
# ### ==========================ff===================================================
##        
    np.random.seed()
#    
    #trajectory plot-------------------------
    # from mpl_toolkits.mplot3d import Axes3D    
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1, 3, 1, projection='3d')

    xs=State_init[:,0]
    ys=State_init[:,1]
    zs=State_init[:,2]
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_title('Trajectory of the true model')
#---------------------------------------------------------------
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax = fig.gca(projection='3d')
    xs=State_train[:,0]
    ys=State_train[:,1]
    zs=State_train[:,2]
    ax.plot(xs, ys, zs, lw=0.5) 
    ax.set_title('Trajectory of the learned model using BiNN SDE')
##
    ax = fig.add_subplot(1, 3,3, projection='3d')
    ax = fig.gca(projection='3d')
    xs=State_train_class[:,0]
    ys=State_train_class[:,1]
    zs=State_train_class[:,2]
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_title('Trajectory of the learned model using BiNN')
    fig.savefig('result_SL/traj%i.png' %k)
#    
    # save result in folder result_SL

##  BiNN SDE
    np.save('result_SL/State_init_T%i.npy' %k,State_init) 
    np.save('result_SL/State_train_T%i.npy'% k,State_train)  
# #=============================================================
    np.save('result_SL/RMSE_BiNN_sde_F.npy',RMSE_BiNN_sde_F) 

    np.save('result_SL/RMSE_BiNN_sde_L.npy',RMSE_BiNN_sde_L)       
# =============================================================================
    np.save('result_SL/F_sde_all.npy',F_sde_all)
    np.save('result_SL/L_B_sde_all.npy',L_B_sde_all)
    np.save('result_SL/L_W_sde_all.npy',L_W_sde_all)
    pred_model.save('result_SL/pred_model%i' %k)

# =============================================================================
# ##  BiNN
    np.save('result_SL/State_train_class_T%i.npy'% k,State_train_class)  
    np.save('result_SL/F_class_all.npy',F_class_all)

# # #=============================================================
    np.save('result_SL/RMSE_BiNN_class.npy',RMSE_BiNN_class)
    pred_model.save('result_SL/pred_model_class%i' %k)

    
 