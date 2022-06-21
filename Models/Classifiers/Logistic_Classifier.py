import numpy as np
from ..Layers import *

class Logisitic_Classifier():
    def __init__(self, input_dim, reg = 0.01):
        self.reg = reg
        self.input_dim = input_dim
        self.params = {}
        # Now we do intialization of the weights
        np.random.seed(0)
        self.params['W'] = np.random.randn(input_dim, 1)*0.01
        self.params['b'] = 0
        pass
    def loss(self,X,y = None): # this function will perform both the backward and forward passes and return the gradients 
        # define a mode here, i.e. a training mode or a test mode
        mode = 'test' if y is None else 'train'
        if(mode == 'train'):
            cache = {}
            loss = 0
            grads = {}
            scores, cache['sig'] = sigmoid_forward(X,self.params['W'],self.params['b'])
            loss, cache['log'] = cross_entropy_loss(scores, y)
            # dscores  = cross_entropy_loss_backward(cache['log'])
            dw,db = cross_entropy_and_sigmoid_backward(cache['sig'], y)
            
            loss = loss + self.reg * np.sum(self.params['W'] * self.params['W'])
            grads['W'] = dw + 2 * self.reg * self.params['W']
            grads['b'] = db

            return loss, grads
        else:
            scores, cache['sig'] = sigmoid_forward(X,self.params['W'],self.params['b'])
            loss, cache['log'] = cross_entropy_loss(scores, y)
            loss = loss + self.reg * np.sum(self.params['W'] * self.params['W'])
            return loss
    def predict(self, X):
        scores = sigmoid(np.dot(X, self.params['W']) + self.params['b'])
        return np.round(scores)

            
