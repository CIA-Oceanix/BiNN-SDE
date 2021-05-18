# -*- coding: utf-8 -*-
# Learning algorithm for F et L
# Integration scheme Euler Maruyama 
"""
Created on Wed Jul  8 21:38:26 2020

@author: noura
"""


import numpy as np
from tensorflow.keras.layers import concatenate, add, multiply, Input, Dense,Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


lamda=0.1
def dynamic_model(GD, train,trainync, params,trans_model=None):
    



    def lik_loss(M,residual_f,eps,y_true,inputs):
        batch = tf.shape(M)[0]
        ynew=inputs+(GD.dt_integration*residual_f)
        M_trans=tf.transpose(M,perm=[0,2,1])
        cov_m=tf.math.scalar_mul(eps,tf.matmul(M,M_trans))+tf.math.scalar_mul(lamda,tf.eye(3))
        inv_cov_m=tf.linalg.inv(cov_m)
        inptmp=tf.reshape((y_true-ynew),shape=(batch,1,params['dim_stochastic']))
        inptmp_trans=tf.transpose(inptmp,perm=[0,2,1])
        prod1=tf.matmul(inptmp,inv_cov_m)
        prod2=tf.matmul(prod1,inptmp_trans)
        recons_loss=tf.reduce_mean(prod2)+tf.reduce_sum(tf.linalg.det(cov_m))

        return recons_loss


    def custom_loss(M,residual_f,eps,inputs):
        def loss(y_true,y_pred):
            return lik_loss(M,residual_f,eps,y_true,inputs)
        return loss



#--------------------------------------------------------------------------
    outputs = params['dim_stochastic']
    bilin_output = 1
    lin_output=params['dim_stochastic']
        #Euler maruyama weights
    w=[GD.dt_integration*np.eye(params['dim_stochastic'])]
    w.append(np.zeros([params['dim_stochastic']]))
#    ----------------------------------------------------------
    wnoise=[np.sqrt(GD.dt_integration)*np.eye(params['dim_stochastic'])]
    wnoise.append(np.zeros([params['dim_stochastic']]))
##    ----------------------------------------
    w3=[np.eye(params['dim_stochastic'])]
    w3.append(np.zeros([params['dim_stochastic']]))
#    ----------------------------------------------
    inputs=Input(shape=(params['dim_stochastic'],))
 
   # Sampling layer to generate the noise
    def sampling(inp):
        batch = tf.shape(inp)[0]
        epsilon = tf.random.normal(shape=(batch,params['dim_stochastic'],1),mean=0., stddev=1, dtype=tf.float32)
        return tf.squeeze(tf.matmul(inp,epsilon),axis=2)
    
    
    #dynamical model 
 
#    Layers for the deterministic opertator F
#        Bilinear layers (6)
    blin_cells1_1=[]
    blin_cells1_2=[]
    for i in range(0,params['bi_linear_layers']):
        blin_cells1_1.append(Dense(bilin_output, activation='linear',name = 'biLin_cell1_1'+str(i))(inputs))
        blin_cells1_2.append(Dense(bilin_output, activation='linear',name = 'biLin_cell1_2'+str(i))(inputs))
#        Linear layer 
    lin_cell = Dense(lin_output, activation='linear', name = 'linear_cell')(inputs)
    # Bilinear terms
    blin_terms=[]
    for i in range(0,params['bi_linear_layers']):
        blin_terms.append(multiply([blin_cells1_1[i], blin_cells1_2[i]],name = 'biLin_products'+str(i)))
        
    layers_concat = lin_cell
    for i in range(0,params['bi_linear_layers']):
        layers_concat=concatenate([blin_terms[i], layers_concat],  name = 'terms_concatenation'+str(i))
    
#       Residual layers
    residual_layers=[]
    residual_layers.append(layers_concat)
    for i in range(0,params['transition_layers']):
        residual_layers.append(Dense(outputs, activation='linear',name = 'residual_layer'+str(i))(residual_layers[i]))
        
    residual_f = Dense(outputs, activation='linear',name='residual_computation')(residual_layers[-1])
#    Layers for the Stochastic opertator L
   
    tmp=Dense(lin_output, activation='linear', name='tensor_0')(inputs)
    tensor_0=tf.linalg.diag(tmp)
    lin_cell_L = Lambda(sampling)(tensor_0)


   # Integration scheme using Euler Maruyama 
   # determinitic component
    RK_K1= Dense(outputs, activation='linear',name='RK_K1',weights=w,trainable=False)(residual_f)
    
   # stochastic component
    RK_K2= Dense(outputs, activation='linear',name='RK_K2',weights=wnoise,trainable=False)(lin_cell_L)


    RK_sum=add([RK_K1, RK_K2],name='RK_sum')
    RK_coef= Dense(outputs, activation='linear',name='RK_Coef',weights=w3,trainable=False)(RK_sum)
    output_prediction=add([RK_coef ,inputs], name='prediction')   
    pred_model =[Model(inputs, output_prediction)]

    
    
    pred_model[-1].compile(optimizer='adam',loss=custom_loss(tensor_0,residual_f,GD.dt_integration,inputs))
    pred_model[-1].fit(train,trainync,batch_size=params['batch_size'],epochs=params['ntrain'][1])#,callbacks=ES
#    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', mode='auto',factor=0.2,
#                              patience=5, min_lr=0.001)
#    pred_model[-1].fit(train,trainync,batch_size=params['batch_size'],epochs=params['ntrain'][1], callbacks=[reduce_lr])

#    return  pred_model[0]
    return  pred_model[0]
    

def get_layer_IDX(model,layer_name):
    layer = np.nan
    for i in range(0,len(model.layers)):
        if (model.layers[i].name==layer_name):
            layer = i
    assert (layer != np.nan),"Layer not found !"
    return layer



