import sympy as sp
import numpy as np

def get_dynamical_model(Dyn_model,model_dim,blin_product,transition_layers):
#    Dyn_model : keras dynamical model F where dot(x)=F(x)
#    blin_product : number of bilinear terms in the dynamical model 
#    model_dim : dimension of the state vector x 
#    transition_layers : number of transition layers in the dyn_model (without counting the regression layer)

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x1x2, x1x3, x2x3 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x1x2 x1x3 x2x3')
    states=np.array([x1, x2, x3])
    lin_inp=np.dot(np.transpose(Dyn_model.layers[get_layer_IDX(Dyn_model,'linear_cell')].get_weights()[0]),states)+Dyn_model.layers[get_layer_IDX(Dyn_model,'linear_cell')].get_weights()[1]
    blin_inp_1=[]
    blin_inp_2=[]
    blin_inp=[]

    layers_concat=[]
    for i in range(lin_inp.shape[0]-1,-1,-1):
        layers_concat.append(lin_inp[i])
            
    for i in range(1,blin_product+1):
        blin_inp_1.append(np.dot(np.transpose(Dyn_model.layers[get_layer_IDX(Dyn_model,'biLin_cell1_1'+str(i-1))].get_weights()[0]),states)+Dyn_model.layers[get_layer_IDX(Dyn_model,'biLin_cell1_1'+str(i-1))].get_weights()[1])
        blin_inp_2.append(np.dot(np.transpose(Dyn_model.layers[get_layer_IDX(Dyn_model,'biLin_cell1_2'+str(i-1))].get_weights()[0]),states)+Dyn_model.layers[get_layer_IDX(Dyn_model,'biLin_cell1_2'+str(i-1))].get_weights()[1])
        blin_inp.append(sp.expand(blin_inp_1[i-1][0]*blin_inp_2[i-1][0]))

    for i in range(1,blin_product+1):
        layers_concat.append(np.reshape(np.array(blin_inp[i-1]),(1)))
    aug_states_bi_nn = np.array(layers_concat)[::-1]
    residual_layer=[aug_states_bi_nn]
    for i in range(1,transition_layers+1):
        residual_layer.append(np.dot(np.transpose(Dyn_model.layers[get_layer_IDX(Dyn_model,'residual_layer0')].get_weights()[0]),residual_layer[-1])+Dyn_model.layers[get_layer_IDX(Dyn_model,'residual_layer'+str(i-1))].get_weights()[1])
    residual= (np.dot(np.transpose(Dyn_model.layers[get_layer_IDX(Dyn_model,'residual_computation')].get_weights()[0]),residual_layer[-1])+Dyn_model.layers[get_layer_IDX(Dyn_model,'residual_computation')].get_weights()[1])

    return residual

def get_dynamical_model_aug_states(Dyn_model,model_dim,blin_product):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y1, y2, y3 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 y1 y2 y3')
    states2=np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
    states=np.array([y1, y2, y3])

    blin_inp_1=[]
    blin_inp_2=[]
    blin_inp=[]
    layers_concat=[]
    
    states_red=np.dot(np.transpose(Dyn_model.layers[get_layer_IDX(Dyn_model,'input_red')].get_weights()[0]),states2)+Dyn_model.layers[get_layer_IDX(Dyn_model,'input_red')].get_weights()[1]
    lin_inp=np.dot(np.transpose(Dyn_model.layers[get_layer_IDX(Dyn_model,'linear_cell')].get_weights()[0]),states)+Dyn_model.layers[get_layer_IDX(Dyn_model,'linear_cell')].get_weights()[1]
        
    for i in range(lin_inp.shape[0]-1,-1,-1):
        layers_concat.append(lin_inp[i])
            
    for i in range(1,blin_product+1):
        blin_inp_1.append(np.dot(np.transpose(Dyn_model.layers[get_layer_IDX(Dyn_model,'biLin_cell1_1'+str(i-1))].get_weights()[0]),states)+Dyn_model.layers[get_layer_IDX(Dyn_model,'biLin_cell1_1'+str(i-1))].get_weights()[1])
        blin_inp_2.append(np.dot(np.transpose(Dyn_model.layers[get_layer_IDX(Dyn_model,'biLin_cell1_2'+str(i-1))].get_weights()[0]),states)+Dyn_model.layers[get_layer_IDX(Dyn_model,'biLin_cell1_2'+str(i-1))].get_weights()[1])
        blin_inp.append(sp.expand(blin_inp_1[i-1][0]*blin_inp_2[i-1][0]))

    for i in range(1,blin_product+1):
        layers_concat.append(np.reshape(np.array(blin_inp[i-1]),(1)))
    aug_states_bi_nn = np.array(layers_concat)[::-1]

    residual_layer= np.dot(np.transpose(Dyn_model.layers[get_layer_IDX(Dyn_model,'residual_layer0')].get_weights()[0]),aug_states_bi_nn)+Dyn_model.layers[get_layer_IDX(Dyn_model,'residual_layer0')].get_weights()[1]
    residual= (np.dot(np.transpose(Dyn_model.layers[get_layer_IDX(Dyn_model,'residual_computation')].get_weights()[0]),residual_layer)+Dyn_model.layers[get_layer_IDX(Dyn_model,'residual_computation')].get_weights()[1])

    return residual

def get_layer_IDX(model,layer_name):
    layer = np.nan
    for i in range(0,len(model.layers)):
        if (model.layers[i].name==layer_name):
            layer = i
    assert (layer != np.nan),"Layer not found !"
    return layer


