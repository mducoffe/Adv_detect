#### second annexe for real fisher matrices : convolutional biases and batch normalization #####
import theano
import theano.tensor as T
import re
from collections import OrderedDict
import keras.backend as K


def get_biases(model, type_layer=['conv']):
    layers = model.layers
    dico = {}
    for layer in layers:
        if (re.match("convolution(.?)(.?)_(.*)", layer.name) and ('conv' in type_layer)):
            number = (int)(layer.name.split('_')[-1])
            b = layer.b
            dico['conv_'+str(number)+'_bias'] = b
        if (re.match("batchnormalization_(.*)", layer.name) and ('batch_norm' in type_layer)):
            number = (int)(layer.name.split('_')[-1])
            beta = layer.beta
            gamma = layer.gamma
            dico['batchnormalization_'+str(number)]=[gamma, beta]
    return dico

def fisher_info(grad_biases):
   for key in grad_biases:
       if re.match('conv_(.*)_bias', key):
           tmp = grad_biases[key].dimshuffle((0, 'x'))
           grad_biases[key] = T.dot(tmp, tmp.T)
       if re.match('batchnormalization_(.*)', key):
           tmp_beta = T.flatten(grad_biases[key][0])
           tmp_gamma = T.flatten(grad_biases[key][1])
           tmp = T.concatenate([tmp_beta, tmp_gamma], axis=0)
           tmp = tmp.dimshuffle((0, 'x'))
           grad_biases[key] = T.dot(tmp, tmp.T)/tmp.norm(2)
   return grad_biases


def build_biases(model):
    # build cost function and computational
    x = T.tensor4(); y = T.imatrix();
    #cost = T.mean(T.nnet.categorical_crossentropy(model.call(x), y.flatten()))
    cost = K.mean(K.categorical_crossentropy(model.call(x), y.flatten()))
    
    # retrieve convolutional biases
    dico_fisher = get_biases(model, type_layer=['conv', 'batch_norm'])
    grad_biases = OrderedDict([(key, K.gradients(cost, dico_fisher[key])) for key in dico_fisher])
    
    return theano.function([x, y], fisher_info(grad_biases), allow_input_downcast=True, on_unused_input='ignore')

def build_fisher_biases(X, Y, model, batch_size=200):
    dico = None 
    if len(X) < batch_size:
        batch_size = len(X)
    f = build_biases(model)
    for minibatch in range(len(Y)/batch_size):
    
        x_batch = X[minibatch*batch_size:(minibatch+1)*batch_size]
        y_batch = Y[minibatch*batch_size:(minibatch+1)*batch_size]
        dico_batch = f(x_batch, y_batch)

        if dico is None:
            dico = dico_batch
        else:
            for key in dico.keys():
                dico[key] += dico_batch[key]
        
        del dico_batch
        return dico
    
    # if len(Y) is not a multiple of batch_size, still take in account the last samples
    if len(Y) % batch_size !=0:
        minibatch = len(Y)/batch_size
        x_batch = X[minibatch*batch_size:]
        y_batch = Y[minibatch*batch_size:]
        dico_batch = f(x_batch, y_batch)
        if dico is None:
            dico = dico_batch
        else:
            for key in dico.keys():
                dico[key] += dico_batch[key]
        
        del dico_batch
    
    for key in dico:
        dico[key]/=(1.*len(Y)/batch_size)
    
    return dico
