import numpy as np
from ..Layers import *

class TwoLayerNet():
    def __init__(self, input_dim, hidden_dim, output_dim, reg = 0.01, weight_scale = 1e-3):
        self.reg = reg
        self.params = {}
        # Now we do intialization of the weights
        np.random.seed(0)
        self.params['W1'] = np.random.randn(input_dim, hidden_dim)*weight_scale#np.sqrt(2.0/input_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, output_dim)*weight_scale#np.sqrt(2.0/hidden_dim)
        self.params['b2'] = np.zeros(output_dim)
        self.output_dim = output_dim

    def loss(self,X,y = None):
        # define a mode here, i.e. a training mode or a test mode
        mode = 'test' if y is None else 'train'
        if(mode == 'train'):
            cache = {}
            loss = 0
            grads = {}
            z1, cache['affine_1'] = affine_forward(X,self.params['W1'],self.params['b1'])
            a1,cache['relu_1'] = relu_forward(z1)
            a2,cache['affine_2'] = affine_forward(a1,self.params['W2'], self.params['b2'])
            loss, da2 = softmax_loss(a2,y)
            da1, grads['W2'],grads['b2'] = affine_backward(da2,cache['affine_2'])
            dz1 = relu_backward(da1, cache['relu_1'])
            dx, grads['W1'],grads['b1'] = affine_backward(dz1, cache['affine_1'])
            loss = loss + 0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
            grads['W1'] = grads['W1'] + 1 * self.reg * self.params['W1']
            grads['W2'] = grads['W2'] + 1 * self.reg * self.params['W2']
            return loss, grads
        else:
            cache = {}
            loss = 0
            z1, cache['affine_1'] = affine_forward(X,self.params['W1'],self.params['b1'])
            a1,cache['relu_1'] = relu_forward(z1)
            a2,cache['affine_2'] = affine_forward(a1,self.params['W2'], self.params['b2'])
            return a2
            # loss, da2 = softmax_loss(a2,y)
            # loss = loss + self.reg * (np.sum( (self.params['W1'] * self.params['W1'])))
            # loss = loss + self.reg * (np.sum( (self.params['W2'] * self.params['W2'])))
            # return loss
    def predict(self, X):
        cache= {}
        z1, cache['affine_1'] = affine_forward(X,self.params['W1'],self.params['b1'])
        a1,cache['relu_1'] = relu_forward(z1)
        a2,cache['affine_2'] = affine_forward(a1,self.params['W2'], self.params['b2'])
        print(a2.shape)
        return np.argmax(a2,axis = 1)



