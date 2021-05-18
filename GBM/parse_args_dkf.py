"""
Parse command line and store result in params
Model : DKF
"""
import argparse,copy
from collections import OrderedDict
p = argparse.ArgumentParser(description="Arguments for variational autoencoder")
parser = argparse.ArgumentParser()
parser.add_argument('-dset','--dataset', action='store',default = '', help='Dataset', type=str)
parser.add_argument('-f')


parser.add_argument('-ds','--dim_stochastic', action='store',default = 3, help='Stochastic dimensions', type=int)
parser.add_argument('-dh','--dim_hidden', action='store', default = 2, help='Hidden dimensions in DKF', type=int)
parser.add_argument('-tl','--transition_layers', action='store', default = 2, help='Layers in transition fxn', type=int)
parser.add_argument('-ttype','--transition_type', action='store', default = 'bi_linear_MLP', help='Layers in transition fxn', type=str, choices=['mlp','simple_gated'])
parser.add_argument('-nt','--ntrain', action='store',type=int,default=5000,help='number of training')
parser.add_argument('-do','--dim_observations', action='store',default = 3, help='observations dimensions', type=int)
parser.add_argument('-bll','--bi_linear_layers', action='store',default = 3, help='number of bilinear layers', type=int)


params = vars(parser.parse_args())

hmap       = OrderedDict() 

hmap['dim_hidden']='dh'
hmap['dim_stochastic']='ds'
hmap['batch_size']='bs'
hmap['epochs']='ep'
hmap['rnn_size']='rs'
hmap['transition_type']='ttype'
hmap['ntrain']='nt'
combined   = ''
for k in hmap:
    if k in params:
        if type(params[k]) is float:
            combined+=hmap[k]+'-'+('%.4e')%(params[k])+'-'
        else:
            combined+=hmap[k]+'-'+str(params[k])+'-'


"""
import cPickle as pickle
with open('default.pkl','wb') as f:
    pickle.dump(params,f)
"""
