import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict  # this can be moved to the top with the other imports
import re
import keras.backend as K


# pure python function
def build_f_l(model, model_input, model_output, dico_filters):
	func = None
	f = build_intermediate_var(model, model_input, model_output)
	def func(X,Y):
		dico = f(X,Y)
		return preprocessing_blocks(dico, dico_filters)
	return func

# GPU dot product
# theano function
def mul_theano():
    M = K.theano.tensor.tensor3()
    def f(index, M):
        return T.dot(M[index].transpose(), M[index])
    output,_ = K.theano.scan(fn=f, outputs_info=None,
				      sequences=[K.T.arange(M.shape[0])],
				      non_sequences=[M])
    g = theano.function([M], output, allow_input_downcast=True)
    def function(matrices, memory_optim=False, split=1000):
        if matrices.ndim==2:
            matrices = matrices.reshape((1, matrices.shape[0], matrices.shape[1]))
            N = matrices.shape[0]
            if N >split and memory_optim:
                n_split = [0]+[split*(i+1) for i in range(N/split)]
                if n_split[-1] >N:
                    n_split[-1] = N
                if n_split[-1]<N:
                    n_split.append(N)
                return sum([g(matrices[n_split[i]: n_split[i+1], :,:]) for i in range(len(n_split) -1)])
            return g(matrices)
        else:
            return g(matrices)
    return function


# TO DO integrer 1/sqrt(tau)
# pure python function
def expansion_op(A, delta):
	# shape = (M, J, X, Y)
	#A = A.transpose((0,3, 1,2))
	(M, J, X, Y) = A.shape
	d_x = delta[0]/2; d_y = delta[1]/2
	E_A = np.zeros((M,X -2*d_x, Y - 2*d_y, J, 2*d_x+1, 2*d_y+1), dtype=A.dtype)
	for n_x in range(d_x, X -d_x):
		for n_y in range(d_y, Y-d_y):
			tmp = A[:,:, n_x -d_x:n_x+d_x+1, n_y-d_y:n_y+d_y+1]
			E_A[:,n_x-d_x, n_y-d_y,:] = tmp[:,:, ::-1, ::-1]
            
	"""
    for m in range(M):
		for j in range(J):
			for n_x in range(d_x, X -d_x):
				for n_y in range(d_y, Y-d_y):
					E_A[m,n_x-d_x, n_y-d_y, j] = np.flipud(np.fliplr(A[m,j, n_x -d_x:n_x+d_x+1, n_y-d_y:n_y+d_y+1]))
    """
	coeff = np.sqrt(1./((X-2*d_x)*(Y-2*d_y))) # 1/sqrt(tau)
	E_A = E_A.reshape((M, (X-2*d_x)*(Y-2*d_y), J*(2*d_x+1)*(2*d_y+1)))
	return coeff*E_A
"""
def expansion_op(A, delta):
        #A = A.transpose((3,2,1,0))
        A = A.transpose((0,3, 1,2)) # nb filter must be in the second position
        
	# shape = (M, J, X, Y)
	(M, J, X, Y) = A.shape
	d_x = delta[0]/2; d_y = delta[1]/2
	E_A = np.zeros((M,X -2*d_x, Y - 2*d_y, J, 2*d_x+1, 2*d_y+1), dtype=A.dtype)
	for m in range(M):
		for j in range(J):
			for n_x in range(d_x, X -d_x):
				for n_y in range(d_y, Y-d_y):
					E_A[m,n_x-d_x, n_y-d_y, j] = np.flipud(np.fliplr(A[m,j, n_x -d_x:n_x+d_x+1, n_y-d_y:n_y+d_y+1]))

	coeff = np.sqrt(1./((X-2*d_x)*(Y-2*d_y))) # 1/sqrt(tau)
	E_A = E_A.reshape((M,(X-2*d_x)*(Y-2*d_y), J*(2*d_x+1)*(2*d_y+1)))
	Id = np.diag([1]*((X-2*d_x)*(Y-2*d_y)))
	E_B = np.zeros((M,(X-2*d_x)*(Y-2*d_y), J*(2*d_x+1)*(2*d_y+1) + (X-2*d_x)*(Y-2*d_y)))
	for m in range(M):
		E_B[m] = np.concatenate([E_A[m], Id], axis=1)
	#import pdb
	#pdb.set_trace()
	#E_A = np.concatenate([E_A, Id], axis=1)
	return E_B
"""	

def preprocessing_blocks(dico, dico_filters):
    for key in dico.keys():
        if re.match('conv_(.*)_input', key):
            dico[key]=expansion_op(dico[key],dico_filters[key])
    return dico


def build_dict_variables(layer_names, input_tensors, output_tensors):
    dico = {'conv':{}, 'dense':{}}

    for layer_name, input_T, output_T in zip(layer_names, input_tensors, output_tensors):

        if re.match("convolution(.?)(.?)_(.*)", layer_name):

            if input_T.ndim ==2 or output_T.ndim==2:
                print('oups')
                import pdb; pdb.set_trace()
            number = (int)(layer_name.split('_')[-1])
            input_conv = input_T
            #input_conv = layer.get_input_at(0)
            output_conv = output_T
            #output_conv = layer.get_output_at(0)
            dico['conv'][number]=[input_conv, output_conv]

        if re.match("dense_(.*)", layer_name):
            number = (int)(layer_name.split('_')[-1])
            input_fully = input_T
            #input_fully = layer.get_input_at(0)
            output_fully = output_T
            #output_fully = layer.get_output_at(0)
            dico['dense'][number]=[input_fully, output_fully]

    return dico


def build_dictionnary(cost, layer_names, input_tensors, output_tensors):

	dico = build_dict_variables(layer_names, input_tensors, output_tensors)
    # preprocessing for dense layers
	dico_dense = dico['dense']
	for key in dico_dense:
            # because we a*aT instead of aT*a we took directly the transpose
            [grad_s] = K.T.grad(cost, [dico_dense[key][1]])
            dico_dense[key][1] = grad_s #TEST
            var_input = dico_dense[key][0]
            var_input = K.T.concatenate([var_input, K.T.ones((var_input.shape[0], 1))], axis=1)
            dico_dense[key][0] = var_input
            dico['dense'] = dico_dense  
    # preprocessing for convolutional layers
	dico_conv = dico['conv']
	for key in dico_conv:
            [grad_s] = K.gradients(cost, [dico_conv[key][1]])
            if grad_s.ndim !=4:
                import pdb; pdb.set_trace()
            grad_s = grad_s.dimshuffle((0, 3, 1, 2)) # ???
            shape = grad_s.shape
            grad_s = grad_s.reshape((shape[0], shape[1], shape[2]*shape[3]))
            grad_s = grad_s.dimshuffle((0,2,1)) # transpose to satisfy Roger inequality
            dico_conv[key][1] = grad_s
            dico_conv[key][0] = dico_conv[key][0].dimshuffle((0, 3, 1, 2))
            dico['conv'] = dico_conv
        
	dico_final = {}
	for key in dico['dense']:
            dico_final['dense_'+str(key)+'_input']=dico['dense'][key][0]
            dico_final['dense_'+str(key)+'_output']=dico['dense'][key][1]
        
	for key in dico['conv']:
            dico_final['conv_'+str(key)+'_input']=dico['conv'][key][0]
            dico_final['conv_'+str(key)+'_output']=dico['conv'][key][1]
        
	return OrderedDict(dico_final)

def build_intermediate_var(model, model_input, model_output):
    # build cost function and computational
    x = K.T.tensor4(); y = K.T.imatrix();
    #cost = T.mean(T.nnet.categorical_crossentropy(model.call(x), y.flatten()))
    #cost = T.mean(model.output)
    input_tensors = model_input.call(x)
    output_tensors = model_output.call(x)
    #cost = T.mean(T.nnet.categorical_crossentropy(output_tensors[-1], y.flatten()))

    cost = K.mean(K.categorical_crossentropy(output_tensors[-1], y.flatten()))
    dico = build_dictionnary(cost, [layer.name for layer in model.layers], input_tensors, output_tensors)
    
    return K.theano.function([x,y], dico, allow_input_downcast=True, on_unused_input='ignore')

def build_filters(model):
    dico = {}
    layers = model.layers
    for layer in layers:
       if re.match("convolution(.?)(.?)_(.*)", layer.name):
            number = (int)(layer.name.split('_')[-1])
            dico['conv_'+str(number)+'_input'] = [layer.nb_col, layer.nb_row]

    return dico

def build_fisher(X, Y, model, model_input, model_output, batch_size=512, f=None):
    print('start')
    dico_filters = build_filters(model)
    if f is None:
        f = build_f_l(model, model_input, model_output, dico_filters)
    print('f exist')
    dico = None
    g = mul_theano()
    if len(X) < batch_size:
		batch_size = len(X)
    n = len(Y)/batch_size
    for minibatch in range(n):
        print((minibatch, n))
        x_batch = X[minibatch*batch_size:(minibatch+1)*batch_size]
        y_batch = Y[minibatch*batch_size:(minibatch+1)*batch_size]
        dico_batch = f(x_batch, y_batch) # theano function + preprocessing blocks
        if dico is None:
            dico = {}
            for key in dico_batch.keys():
                var = dico_batch[key]
                if re.match('conv_(.*)_input', key):
                    shape = var.shape
                    var = var.reshape((shape[0]*shape[1], shape[2]))
                    tmp = g(var, True)
                else:
                    tmp = g(var, False)
                assert tmp.ndim==3 and (tmp.shape[1]==tmp.shape[2]),'wrong dimensions for covariance matrices'
                dico[key]=np.sum(tmp, axis=0)

        else:
            for key in dico_batch.keys():
                var = dico_batch[key]
                if re.match('conv_(.*)_input', key):
                    shape = var.shape
                    var = var.reshape((shape[0]*shape[1], shape[2]))
                    tmp = g(var, True)
                else:
                    tmp = g(var, False)
                assert tmp.ndim==3 and (tmp.shape[1]==tmp.shape[2]),'wrong dimensions for covariance matrices'
                dico[key]+=np.sum(tmp, axis=0)
        del dico_batch

	# if len(Y) is not a multiple of batch_size, still take in account the last samples
	if len(Y) % batch_size !=0:
		minibatch = len(Y)/batch_size
		x_batch = X[minibatch*batch_size:]
		y_batch = Y[minibatch*batch_size:]
		dico_batch = f(x_batch, y_batch)
                if dico is None :
                    dico = {}
                    for key in dico_batch.keys():
                        var = dico_batch[key]
                        if re.match('conv_(.*)_input', key):
                            shape = var.shape
                            var = var.reshape((shape[0]*shape[1], shape[2]))
                            tmp = g(var, True)
                        else:
                            tmp = g(var, False)
                        assert tmp.ndim==3 and (tmp.shape[1]==tmp.shape[2]),'wrong dimensions for covariance matrices'
                        dico[key]=np.sum(tmp, axis=0)
                else:
                   for key in dico_batch.keys():
                        var = dico_batch[key]
                        if re.match('conv_(.*)_input', key):
                            shape = var.shape
                            var = var.reshape((shape[0]*shape[1], shape[2]))
                            tmp = g(var, True)
                        else:
                            tmp = g(var, False)
                        assert tmp.ndim==3 and (tmp.shape[1]==tmp.shape[2]),'wrong dimensions for covariance matrices'
                        dico[key]+=np.sum(tmp, axis=0)
		del dico_batch
    for key in dico.keys():
        dico[key]/=(1.*len(X))
    """
    for key in dico.keys():
        shape = dico[key].shape[0]
        dico[key]+=1e-8*np.diag(np.ones(shape))
    """
    return dico, f

def build_queries(X, Y, model, model_input, model_output, batch_size=512):
    dico_filters = build_filters(model)
    f = build_f_l(model, model_input, model_output, dico_filters)
    dico = None
    g = mul_theano()
    if len(X) < batch_size:
		batch_size = len(X)
    n = len(Y)/batch_size
    for minibatch in range(n):
        print((minibatch, n))
        x_batch = X[minibatch*batch_size:(minibatch+1)*batch_size]
        y_batch = Y[minibatch*batch_size:(minibatch+1)*batch_size]
        dico_batch = f(x_batch, y_batch) # theano function + preprocessing blocks
        if dico is None:
            dico = {}
            for key in dico_batch.keys():
                var = dico_batch[key]
                print((key, var.shape))
                if re.match('conv_(.*)_input', key):
                    tmp = g(var, False)
                else:
                    tmp = g(var, False)
                assert tmp.ndim==3 and (tmp.shape[1]==tmp.shape[2]),'wrong dimensions for covariance matrices'
                dico[key]=tmp
                print((key, len(y_batch), tmp.shape))

        else:
            for key in dico_batch.keys():
                var = dico_batch[key]
                if re.match('conv_(.*)_input', key):
                    tmp = g(var, True)
                else:
                    tmp = g(var, False)
                assert tmp.ndim==3 and (tmp.shape[1]==tmp.shape[2]),'wrong dimensions for covariance matrices'
                dico[key]= np.concatenate([dico[key], tmp], axis=0)

        del dico_batch
	# if len(Y) is not a multiple of batch_size, still take in account the last samples
	if len(Y) % batch_size !=0:
		minibatch = len(Y)/batch_size
		x_batch = X[minibatch*batch_size:]
		y_batch = Y[minibatch*batch_size:]
		dico_batch = f(x_batch, y_batch)
                if dico is None :
                    dico = {}
                    for key in dico_batch.keys():
                        var = dico_batch[key]
                        if re.match('conv_(.*)_input', key):
                            tmp = g(var, True)
                        else:
                            tmp = g(var, False)
                        assert tmp.ndim==3 and (tmp.shape[1]==tmp.shape[2]),'wrong dimensions for covariance matrices'
                        dico[key]=tmp
                else:
                   for key in dico_batch.keys():
                        var = dico_batch[key]
                        if re.match('conv_(.*)_input', key):
                            tmp = g(var, True)
                        else:
                            tmp = g(var, False)
                        assert tmp.ndim==3 and (tmp.shape[1]==tmp.shape[2]),'wrong dimensions for covariance matrices'
                        dico[key]=np.concatenate([dico[key], tmp], axis=0)
		del dico_batch
    """
    for key in dico.keys():
        dico[key]/=(1.*len(X))
    """
    return dico