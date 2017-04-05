"""
computing approximation of the Fisher Information
for CNN : convolution, fully connected and logistic layer
"""

"""
hypothesis structure :
D_P = { psi : {layer_number : psi__mean_value}, {tau : {layer_number : tau_mean_value}}
this is for unlabelled data
for the data in the subset one should record of every data in an independent dictionary :
D_Q = { index_sample : { psi : {layer_number : psi__mean_value}, {tau : {layer_number : tau_mean_value}}}
"""
from keras.models import Model, Sequential
import numpy as np
from annexe import build_fisher, build_queries
from annexe_bis import build_fisher_biases
import re
import pickle as pkl
from contextlib import closing
import os
from sampling import Sampling
from markov_chhain_sampling import Sampling_MC
import keras.backend as K
import theano.tensor as T
import sklearn.metrics as metrics



class Fisher(object):

    def __init__(self, network):
        """
        proper documentation
        """
        # remove dropout from the network if needed
        self.network = network
        self.f = None
        self.stochastic_layers = {}
        self.filter_layer()
        
        layers = self.network.layers
        for layer in layers:
            if layer.name in self.stochastic_layers:
                tmp = self.stochastic_layers[layer.name]
                setattr(layer, tmp[0], tmp[2])
        
        layers = self.network.layers
        intermediate_layers_input = []
        intermediate_layers_output = []
        for layer in layers:
            if re.match('merge_(.*)', layer.name):
                intermediate_layers_input.append(layer.input[0])
                intermediate_layers_output.append(layer.output)
            else:
                intermediate_layers_input.append(layer.input)
                intermediate_layers_output.append(layer.output)

        self.intermediate_input = Model(self.network.input, [input_ for input_ in intermediate_layers_input])
        self.intermediate_output = Model(self.network.input, [output_ for output_ in intermediate_layers_output])
        self.f = None
        layers = self.network.layers
        for layer in layers:
            if layer.name in self.stochastic_layers:
                tmp = self.stochastic_layers[layer.name]
                setattr(layer, tmp[0], tmp[1])
                
        self.dico_fisher = None
        self.sampler = None


    def filter_layer(self):
        layers = self.network.layers
        for layer in layers:
            if re.match('dropout_(.*)', layer.name):
                self.stochastic_layers[layer.name]=['p', layer.p, 0]
            if re.match('batchnormalization_(.*)', layer.name):
                self.stochastic_layers[layer.name]=['mode',layer.mode, 1]


    def fisher_information(self, X, Y, recompute=False):
        if not(recompute) and not(self.dico_fisher is None):
            return self.dico_fisher
        assert X.ndim==4, ("X must be a 4d tensor of shape (num_elem, num_channels, width, height) but has %d dimensions", X.ndim)
        assert Y.ndim<=2, ("Y contains the class label and should be of size (num_elem, 1) but has %d dimensions", Y.ndim)

        if Y.ndim==1:
            #Y = Y.dimshuffle((0, 'x'))
            Y = Y[:,None]

        if Y.ndim==2 and Y.shape[1]>1:
            Y = np.argmax(Y, axis=1)[:,None]
        
        layers = self.network.layers
        for layer in layers:
            if layer.name in self.stochastic_layers:
                tmp = self.stochastic_layers[layer.name]
                setattr(layer, tmp[0], tmp[2])

        dico_fisher, f = build_fisher(X, Y, self.network, self.intermediate_input, self.intermediate_output, f =self.f) # remove break
        self.f = f
        dico_conv_biases = build_fisher_biases(X, Y, self.network)
        for key in dico_conv_biases:
            dico_fisher[key] = dico_conv_biases[key]

        for layer in layers:
            if layer.name in self.stochastic_layers:
                tmp = self.stochastic_layers[layer.name]
                setattr(layer, tmp[0], tmp[1])
                
        # TO DO INVERSE LAYERS !!!!!!!
        self.dico_fisher = dico_fisher
        return self.dico_fisher
    
    def fisher_queries(self, X, Y):

        assert X.ndim==4, ("X must be a 4d tensor of shape (num_elem, num_channels, width, height) but has %d dimensions", X.ndim)
        assert Y.ndim<=2, ("Y contains the class label and should be of size (num_elem, 1) but has %d dimensions", Y.ndim)

        if Y.ndim==1:
            #Y = Y.dimshuffle((0, 'x'))
            Y = Y[:,None]

        if Y.ndim==2 and Y.shape[1]>1:
            Y = np.argmax(Y, axis=1)[:,None]
        
        layers = self.network.layers
        for layer in layers:
            if layer.name in self.stochastic_layers:
                tmp = self.stochastic_layers[layer.name]
                setattr(layer, tmp[0], tmp[2])

        dico_fisher = build_queries(X, Y, self.network, self.intermediate_input, self.intermediate_output) # remove break
        """
        dico_conv_biases = build_fisher_biases(X, Y, self.network)
        for key in dico_conv_biases:
            dico_fisher[key] = dico_conv_biases[key]

        for layer in layers:
            if layer.name in self.stochastic_layers:
                tmp = self.stochastic_layers[layer.name]
                setattr(layer, tmp[0], tmp[1])
                
        # TO DO INVERSE LAYERS !!!!!!!
        """
        return dico_fisher
    
    
    def save(self, repo, filename):
        with closing(open(os.path.join(repo, filename), 'wb')) as f:
            pkl.dump(self.dico_fisher, f, protocol=pkl.HIGHEST_PROTOCOL)
            
    def load(self, repo, filename):
        with closing(open(os.path.join(repo, filename), 'rb')) as f:
            self.dico_fisher = pkl.load(f)
        # temporary
        #self.dico_fisher=  dict([('conv_1_bias', self.dico_fisher['conv_1_bias']), ('conv_2_bias', self.dico_fisher['conv_2_bias'])])
        

    def build_mean(self):
        if self.dico_fisher is None:
            print('you need to compute the fisher information first')
            return {}

        layers = self.network.layers
        dico_mean = {}
        for layer in layers:
            if re.match('convolution(.*)', layer.name):
                nb_layer = layer.name.split('_')[1]
                # attention the bias is separated from the weigths
                W, b = layer.get_weights()
                dico_mean['conv_'+nb_layer] = W.flatten()
                dico_mean['conv_'+nb_layer+'_bias'] = b.flatten()
            elif re.match('dense_(.*)', layer.name):
                W, b = layer.get_weights()
                dico_mean[layer.name]=np.concatenate([W.flatten(), b.flatten()], axis=0)

            elif re.match('batchnormalization_(.*)', layer.name):
                gamma = layer.gamma.get_value()
                beta = layer.beta.get_value()
                dico_mean[layer.name]=np.concatenate([gamma.flatten(), beta.flatten()], axis=0)
        return dico_mean
    
    def copy_weights(self, model, dico_weights):
        layers = model.layers
        for layer in layers:
            if re.match('convolution(.*)input(.*)', layer.name):
                continue
            if re.match('convolution(.*)', layer.name):
                nb_layer = layer.name.split('_')[1]
                
                name_W = 'conv_'+nb_layer
                if name_W in dico_weights.keys():
                    W = layer.W;
                    W_value = dico_weights['conv_'+nb_layer].astype('float32')
                    W_shape = W.shape.eval()
                    W.set_value(W_value.reshape(W_shape))
                
                name_b = 'conv_'+nb_layer+'_bias'
                if name_b in dico_weights.keys():
                    b = layer.b
                    b_value = dico_weights['conv_'+nb_layer+'_bias'].astype('float32')
                    b_shape = b.shape.eval()
                    b.set_value(b_value.reshape(b_shape))
                
            elif re.match('dense_(.*)', layer.name):
                W = layer.W; b = layer.b
                W_shape = layer.W.shape.eval()
                b_shape = layer.b.shape.eval()
                if layer.name in dico_weights.keys():
                    params_W_b = dico_weights[layer.name].astype('float32')
                    split = len(params_W_b) - np.prod(b_shape)
                    W_value = params_W_b[:split]
                    b_value = params_W_b[split:]
                    W.set_value(W_value.reshape(W_shape))
                    b.set_value(b_value.reshape(b_shape))
                
            elif re.match('batchnormalization_(.*)', layer.name):
                gamma = layer.gamma
                beta = layer.beta
                gamma_shape = gamma.shape.eval()
                beta_shape = beta.shape.eval()
                gamma_split = np.prod(gamma_shape)
                if layer.name in dico_weights.keys():
                    params_gamma_beta = dico_weights[layer.name].astype('float32')
                    gamma_value = params_gamma_beta[:gamma_split]
                    beta_value = params_gamma_beta[gamma_split:]
                    gamma.set_value(gamma_value.reshape(gamma_shape))
                    beta.set_value(beta_value.reshape(beta_shape))


    def fisher_sample(self):
        if self.sampler is None:
            print("preprocessing")
            if self.dico_fisher is None:
                print('you need to compute the fisher information first')
                return
            self.sampler = Sampling(self.build_mean(), self.dico_fisher)
            print('sampling ok')
        config = self.network.get_config()
        if self.network.__class__.__name__=='Sequential':
            new_model = Sequential.from_config(config)
        else:
            new_model = Model.from_config(config)
        new_params = self.sampler.sample()
        """
        means = self.sampler.mean
        
        for key in means:
            if np.max(np.abs(means[key] - new_params[key]))==0:
                print key
        print('kikou')
        import pdb; pdb.set_trace()
        """
        #tmp_prob = self.sampler.prob(new_params)
        new_model.compile(loss=self.network.loss,
                          optimizer=str.lower(self.network.optimizer.__class__.__name__),
                          metrics = self.network.metrics)
        new_model.set_weights(self.network.get_weights())
        self.copy_weights(new_model, new_params)
        return new_model
    
    
    """
    def sample_ensemble(self, data, N=2, nb_classe=1):
        (X_train, Y_train), (X_test, Y_test) = data
        def evaluate(model, data_train):
            (X_train,Y_train) = data_train
            yPreds = model.predict(X_train)
            yPred = np.argmax(yPreds, axis=1)
            yTrue = Y_train
    
            return metrics.accuracy_score(yTrue, yPred)

        models = [self.network] + [self.fisher_sample() for i in range(N)]
        probabilities = []
        data_train = (X_train, Y_train)
        import pdb
        for model in models:
            var = evaluate(model, data_train)
            probabilities.append(var)
        #probabilities = [evaluate(model, (X_train, Y_train)) for model in models]
        
        probabilities /= sum(probabilities)

        yPreds_ensemble = np.mean([alpha*model.predict(X_test) for alpha, model in zip(probabilities,models)], axis=0)
        yPred = np.argmax(yPreds_ensemble, axis=1)
        yTrue = Y_test
        accuracy = metrics.accuracy_score(yTrue, yPred) * 100
        print("Accuracy : ", accuracy)
    """
    
    def sample_ensemble(self, data, N=2, nb_classe=1):
        (X_train, Y_train), (X_test, Y_test) = data
        models = [self.fisher_sample() for i in range(N)]
        N-=1
        yPreds_ensemble = np.array([model.predict(X_test) for model in models]) #(N+1, 10000, 10)
        committee = np.mean(yPreds_ensemble, axis=0) # shape(10000, 10)
        def kl_divergence(committee_proba, member_proba):
            # for a fixed i
            # for a fixed j
            yPred=[]
            for n in range(len(Y_test)):
                predict = []
                for i in range(10):
                    proba_i_C = committee_proba[n,i]
                    predict_i=0
                    for j in range(N+1):
                        proba_i_j = member_proba[j,n, i]
                        predict_i += np.log(proba_i_j)*np.log(proba_i_j/proba_i_C)
                    predict.append(predict_i)
                yPred.append(np.argmin(predict))
            return np.array(yPred).astype('uint8')
        yPred = kl_divergence(committee, yPreds_ensemble)
        yTrue = Y_test
        accuracy = metrics.accuracy_score(yTrue, yPred) * 100
        print("Accuracy : ", accuracy)
        
    def correlation_sampling(self, data, N=3, n=5):
        (X_train, Y_train), (X_test, Y_test) = data
        predict_ensemble = [self.network.predict(X_train)]
        ensemble_model = [self.network]
        for l in range(N):
            p_ensemble = np.mean(predict_ensemble, axis=0)
            models = [self.fisher_sample() for p in range(n)]
            predict = [model.predict(X_train) for model in models]
        
            def evaluate(model, data_train):
                (X_train,Y_train) = data_train
                yPreds = model.predict(X_train)
                yPred = np.argmax(yPreds, axis=1)
                yTrue = Y_train
    
                return metrics.accuracy_score(yTrue, yPred)

            def kl(p_network, p_model):
                kl=[]
                n = len(Y_train)
                for i in range(n):
                    kl_n = 0
                    for j in range(10):
                        kl_n += p_network[i,j]*np.log(p_network[i,j]/p_model[i,j])
                        kl.append(kl_n)
                return np.mean(kl)
        
            index = np.argmax([kl(p_ensemble, predict[j]) for j in range(n)])
            model = models[index]
            ensemble_model.append(model)
            predict_ensemble.append(predict[index])
    
        # Accuracy
        #Ypred = np.argmax(np.mean([network.predict(X_test) for network in [self.network, model]], axis=0), axis=1)
        Ypred = np.argmax(np.mean([model.predict(X_test) for model in ensemble_model]))
        Ytrue = Y_test
        print(metrics.accuracy_score(Ytrue, Ypred))
        
    def sample_ensemble(self, data, N=2, nb_classe=1):
        (X_train, Y_train), (X_test, Y_test) = data
        models = [self.network]; scores=[1.]
        for i in range(N):
            model, score = self.fisher_sample()
            models.append(model)
            scores.append(score)
        #models = [self.network] + [self.fisher_sample() for i in range(N)]
        min_prob = np.min(scores)
        scores -= min_prob
        scores = np.exp(scores)
        print(scores)
        """
        def evaluate(model, data_train):
            (X_train,Y_train) = data_train
            yPreds = model.predict(X_train)
            yPred = np.argmax(yPreds, axis=1)
            yTrue = Y_train
    
            return metrics.accuracy_score(yTrue, yPred)
        """
        #probabilities = [evaluate(model, (X_train, Y_train)) for model in models]
        #probabilities /= sum(probabilities)
        #yPreds_ensemble = np.array([alpha*model.predict(X_test) for alpha, model in zip(scores,models)]) #(N+1, 10000, 10)
        #yPreds_network = np.mean(self.network.predict(X_test), axis=0)
        yPred = np.argmax(np.mean(np.array([alpha*model.predict(X_test) for alpha, model in zip(scores,models)]), axis=0), axis=1) #(N+1, 10000, 10)
        """
        yPred = []
        for n in range(len(Y_test)):
            label = np.argmax(yPreds_ensemble[:,n,:]) % (N+1)
            yPred.append(label)

            if label !=Y_test[n]:

                print((np.argmax(yPreds_ensemble[0,n]), label, Y_test[n]))

        """
        yTrue = Y_test
        accuracy = metrics.accuracy_score(yTrue, yPred) * 100
        print("Accuracy : ", accuracy)
        

       
        
                
                






        
            
            
            

            

	   

	    




