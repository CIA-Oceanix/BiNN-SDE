
#Binn model with euler integration scheme, 
import numpy as np
from tensorflow.keras.layers import concatenate, add, multiply, Input, Dense
from tensorflow.keras.models import Model




def dynamic_model_class(GD, train,trainync, params,trans_model=None):
    outputs = params['dim_stochastic']
    bilin_output = 1
    lin_output=params['dim_stochastic']
    
    #Euler weights
    w=[GD.dt_integration*np.eye(params['dim_stochastic'])]
    w.append(np.zeros([params['dim_stochastic']]))
#  
    w3=[np.eye(params['dim_stochastic'])]
    w3.append(np.zeros([params['dim_stochastic']]))
    
    inputs=Input(shape=(params['dim_stochastic'],))
    outputs = params['dim_stochastic']   
    
    #dynamical model 
    
    blin_cells1_1=[]
    blin_cells1_2=[]
    for i in range(0,params['bi_linear_layers']):
        blin_cells1_1.append(Dense(bilin_output, activation='linear',name = 'biLin_cell1_1'+str(i))(inputs))
        blin_cells1_2.append(Dense(bilin_output, activation='linear',name = 'biLin_cell1_2'+str(i))(inputs))
    
    lin_cell = Dense(lin_output, activation='linear', name = 'linear_cell')(inputs)
    
    blin_terms=[]
    for i in range(0,params['bi_linear_layers']):
        blin_terms.append(multiply([blin_cells1_1[i], blin_cells1_2[i]],name = 'biLin_products'+str(i)))
        
    layers_concat = lin_cell
    for i in range(0,params['bi_linear_layers']):
        layers_concat=concatenate([blin_terms[i], layers_concat],  name = 'terms_concatenation'+str(i))
    residual_layers=[]
    residual_layers.append(layers_concat)
    for i in range(0,params['transition_layers']):
        residual_layers.append(Dense(outputs, activation='linear',name = 'residual_layer'+str(i))(residual_layers[i]))
        
    residual = Dense(outputs, activation='linear',name='residual_computation')(residual_layers[-1])
# --------------------------------------------------------------------
# Euler integration scheme
# ---------------------------------------------------------------------
    RK_K1= Dense(outputs, activation='linear',name='RK_K1',weights=w,trainable=False)(residual)
#   
    RK_coef= Dense(outputs, activation='linear',name='RK_Coef',weights=w3,trainable=False)(RK_K1)
    output_prediction=add([RK_coef, inputs], name='prediction')

#   
    
    pred_model =[Model(inputs, output_prediction)]
    pred_model[-1].compile(optimizer='adam', loss='mse')
    pred_model[-1].fit(train,trainync,epochs=params['ntrain'][1])#,callbacks=ES,validation_split=0.4)

    return  pred_model[0]
    

def get_layer_IDX(model,layer_name):
    layer = np.nan
    for i in range(0,len(model.layers)):
        if (model.layers[i].name==layer_name):
            layer = i
    assert (layer != np.nan),"Layer not found !"
    return layer