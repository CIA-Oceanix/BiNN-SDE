# BiNN SDE for GBM
from IPython import get_ipython
get_ipython().magic('reset -sf')


from AnDA_stat_functions import AnDA_RMSE
import numpy as np
import matplotlib.pyplot as plt
from parse_args_dkf import params 
from generate_GM import generate_GM


# parameters
class D_parameters:
    theta = 0.0
    mu = 0.5
#        noise
    gama=1
    # gama=0
    N=3
    dt_integration = 0.001 # integration time
    




Nb_simu=10
F_sde_all=np.zeros((Nb_simu,1))
L_sde_all=np.zeros((Nb_simu,1))
F_class_all=np.zeros((Nb_simu,1))

RMSE_BiNN_sde_F=np.zeros(Nb_simu)
RMSE_BiNN_sde_L=np.zeros(Nb_simu)

RMSE_BiNN_class=np.zeros(Nb_simu)

for k in range(Nb_simu):
    print(k)
# Generate data GBM
    State=generate_GM(D_parameters)
  
    State_init=State
    N=State.shape[0]

#    ----------------------------------------

    trainx=State[0:N-1,:]
    trainync= State[1:N,:]
#-----------------------------------------------------------------
#neural net params 
    params['transition_layers']=1
    params['bi_linear_layers']=0
    params['dim_stochastic']=1
    params['batch_size']=32
    params['ntrain']=[10,2500]
 
    
# # # #------------------------------------------------------------------------
    from BiNN_model_SDE_1D import dynamic_model
    pred_model=dynamic_model(D_parameters,trainx, trainync,params,None)

# =============================================================================
#    History of the model
    hist=pred_model.history.history
# =============================================================================

   

####Generate trajectory using the learned model
    
    State_train=np.zeros(N)
    x0=State_init[0]
    State_train[0]=x0

    for i in range(N-1):
        tmp=np.reshape(x0,(1,))
        ynew=pred_model.predict(tmp)
        State_train[i+1]=ynew[0]
        x0=State_train[i+1]

# # =============================================================================
# # ##    ------------------------------------------------------------------------------


    ##    calculate  L(gama)
    W1=pred_model.get_layer('tensor_0').get_weights()[0].T
    gama=D_parameters.gama

    L_sde_all[k,:]=W1
    RMSE_BiNN_sde_L[k]=AnDA_RMSE(gama,W1)
# ## =============================================================================
# ## Calculate  F(mu)
    a_wb=pred_model.get_layer('linear_cell').get_weights()[0]

    F_sde_all[k,:]=a_wb
    true_weights_wb_sde=np.array([D_parameters.mu])
  
#calcul RMSE
    RMSE_BiNN_sde_F[k]=AnDA_RMSE(true_weights_wb_sde,a_wb)
  
# -------------------------------------------------------------------------
# biNN 

    from Bi_NN_model_1D import dynamic_model_class
    pred_model_class=dynamic_model_class(D_parameters,trainx, trainync,params,None)

    State_train_class=np.zeros(N)
    x0=State_init[0]
    State_train_class[0]=x0

    for i in range(N-1):
    
        tmp=np.reshape(x0,(1,))
        ynew=pred_model_class.predict(tmp)
        State_train_class[i+1]=ynew[0]
        x0=State_train_class[i+1]
        
        
# # # # ## =============================================================================
# ## Calculate  F(mu) 

    a_wb_class=pred_model_class.get_layer('linear_cell').get_weights()[0]
    F_class_all[k,:]=a_wb_class
  
# # #calcul RMSE
    RMSE_BiNN_class[k]=AnDA_RMSE(true_weights_wb_sde,a_wb_class)
# # # # # # # # # #     # --------------------------------------------------

# Save results in the folder result_GBM
# BiNN SDE
    np.save('result_GBM/State_init_T%i.npy' %k,State_init) 
    np.save('result_GBM/State_train_T%i.npy'% k,State_train)  
# #=============================================================
    np.save('result_GBM/RMSE_BiNN_sde_F.npy',RMSE_BiNN_sde_F) 

    np.save('result_GBM/RMSE_BiNN_sde_L.npy',RMSE_BiNN_sde_L)       
    np.save('result_GBM/F_sde_all.npy',F_sde_all)
    np.save('result_GBM/L_sde_all.npy',L_sde_all)
    pred_model.save('result_GBM/pred_model%i' %k)
    # ------------------------------------------------------------------------------
# BiNN
   
    np.save('result_GBM/State_train_class_T%i.npy'% k,State_train_class)  
# #=============================================================
    np.save('result_GBM/RMSE_BiNN_class.npy',RMSE_BiNN_class) 
    np.save('result_GBM/F_class_all.npy',F_class_all)
    pred_model.save('result_GBM/pred_model_class%i' %k)

    