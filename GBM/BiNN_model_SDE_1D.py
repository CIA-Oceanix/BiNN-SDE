# -*- coding: utf-8 -*-
#BiNN SDE
"""
Created on Wed Jul  8 21:38:26 2020

@author: noura
"""

#ajouter euler pour le shema d 'integration
import numpy as np
from tensorflow.keras.layers import  add, Input, Dense,Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()



      
lamda=0
def dynamic_model(GD, train,trainync, params,trans_model=None):
    
    # Custom loss function
    def lik_loss(M,residual_f,eps,y_true,inputs,y_pred):
        
        mu_0=inputs+(GD.dt_integration*residual_f)
        cov_m=M**2
        inv_cov_m=(1/(cov_m+lamda))*(1/eps)
        tmp1=(y_true-mu_0)
     
        prod1=(tmp1**2)*inv_cov_m
        
        reconstruction_losstmp=prod1+(tf.math.log(cov_m+lamda)+tf.math.log(eps))
        reconstruction_loss=tf.reduce_sum(reconstruction_losstmp)  
        # -------------------------------------------------
        return reconstruction_loss


    def custom_loss(M,residual_f,eps,inputs):
        def loss(y_true,y_pred):
            return lik_loss(M,residual_f,eps,y_true,inputs,y_pred)
        return loss
      


#--------------------------------------------------------------------------
    outputs = params['dim_stochastic']
    lin_output=params['dim_stochastic']
        #Euler Maruyama weights
    w=[GD.dt_integration*np.eye(params['dim_stochastic'])]
    w.append(np.zeros([params['dim_stochastic']]))
#    ----------------------------------------------------------
    wnoise=[np.sqrt(GD.dt_integration)*np.eye(params['dim_stochastic'])]
    wnoise.append(np.zeros([params['dim_stochastic']]))
##    ----------------------------------------
    w3=[np.eye(params['dim_stochastic'])]
    w3.append(np.zeros([params['dim_stochastic']]))
    # #-----------------------------------------------------
    wL=[GD.gama*np.eye(params['dim_stochastic'])]
    wL.append(np.zeros([params['dim_stochastic']]))    
#-------------------------------------------------------------------------
    inputs=Input(shape=(params['dim_stochastic'],))
    
    # sampling layer for noise
    
    def sampling(inp):
        batch=tf.shape(inp)[0]
        epsilon = tf.random.normal(shape=(batch,params['dim_stochastic']),mean=0., stddev=1, dtype=tf.float32)
    
        return (inp*epsilon)
    
    
    #dynamical model 
#  layer F

    lin_cell = Dense(lin_output,activation='linear', use_bias=False,kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.),kernel_constraint=tf.keras.constraints.NonNeg(),name = 'linear_cell')(inputs)
    residual_f=lin_cell
###       layer L
    tensor_0=Dense(lin_output,activation='linear',use_bias=False,kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.),kernel_constraint=tf.keras.constraints.NonNeg(),name='tensor_0')(inputs)

    lin_cell_L = Lambda(sampling)(tensor_0) 
    
        

# =============================================================================
# integration scheme
# =============================================================================
    RK_K1= Dense(outputs, activation='linear',name='RK_K1',weights=w,trainable=False)(residual_f)
#    RK_K2_tmp=delta_B*L(xt)

    RK_K2= Dense(outputs, activation='linear',name='RK_K2',weights=wnoise,trainable=False)(lin_cell_L)

    RK_sum=add([RK_K1, RK_K2],name='RK_sum')
    RK_coef= Dense(outputs, activation='linear',name='RK_Coef',weights=w3,trainable=False)(RK_sum)
    output_prediction=add([RK_coef ,inputs], name='prediction')   
 
    
    pred_model =[Model(inputs, output_prediction)]

    pred_model[-1].compile(optimizer='adam',loss=custom_loss(tensor_0,residual_f,GD.dt_integration,inputs))
    pred_model[-1].fit(train,trainync,batch_size=params['batch_size'],epochs=params['ntrain'][1])#,callbacks=ES
   

    return  pred_model[0]
    

