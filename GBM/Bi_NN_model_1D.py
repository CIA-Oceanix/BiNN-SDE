
#Binn algorithm
import numpy as np
from tensorflow.keras.layers import  add, Input, Dense
from tensorflow.keras.models import Model

import tensorflow as tf
# from tensorflow.keras.optimizers import Adam


      
  
def dynamic_model_class(GD, train,trainync, params,trans_model=None):
    
    outputs = params['dim_stochastic']
    lin_output=params['dim_stochastic']
    
    #runge kutta weights
    w=[GD.dt_integration*np.eye(params['dim_stochastic'])]
    w.append(np.zeros([params['dim_stochastic']]))

    w3=[np.eye(params['dim_stochastic'])]
    w3.append(np.zeros([params['dim_stochastic']]))
    
    inputs=Input(shape=(params['dim_stochastic'],))
    outputs = params['dim_stochastic']   
    
    #dynamical model 
     
    lin_cell = Dense(lin_output,  activation='linear',use_bias=False,kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.),kernel_constraint=tf.keras.constraints.NonNeg(),name = 'linear_cell')(inputs)
    
    residual=lin_cell
        
# -------------------------------------------------------------------
    
    #Euler integration scheme
    RK_K1= Dense(outputs, activation='linear',name='RK_K1',weights=w,trainable=False)(residual)
#   
    RK_coef= Dense(outputs, activation='linear',name='RK_Coef',weights=w3,trainable=False)(RK_K1)
    output_prediction=add([RK_coef, inputs], name='prediction')

    pred_model =[Model(inputs, output_prediction)]
    pred_model[-1].compile(optimizer='adam', loss='mse')
    pred_model[-1].fit(train,trainync,epochs=params['ntrain'][1])#,callbacks=ES,validation_split=0.4)

    return  pred_model[0]
    

