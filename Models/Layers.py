import numpy as np

#This is binary cross entropy loss only 
def cross_entropy_loss(a, y):
    # print(a.shape)
    # print(y.shape)
    y_new = y.reshape(a.shape)
    loss_vector = (-1 *( y_new * np.log(a) + (1 - y_new) * np.log(1 - a) ) )
    # print(loss_vector.shape)
    loss = np.sum(loss_vector, axis = 0)
    cache = a, loss_vector, y_new
    # print("loss inside = ", loss)
    return loss, cache


def cross_entropy_loss_backward(cache):
    a, loss_vector, y = cache
    dscores =  ((1 - y) / (1 - a) - (y / a))
    return dscores


def sigmoid(z):
    # print("z min  is = ", np.min(z))
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_forward(x, w, b):
    a = sigmoid(np.dot(x, w) + b)
    cache = (x, w, b, a)
    return a, cache


def sigmoid_backward(dscores, cache): # numeric instability issues with the sigmoid !
    x, w, b, a = cache
    dz = dscores * a * (1 - a)
    dw = np.dot(x.T, dz)
    db = np.sum(dz, axis=0) 
    return dw, db

def cross_entropy_and_sigmoid_backward(cache, y): # utility function for ease
    x, w, b, a = cache
    dz = a - y
    dw = np.dot(x.T, dz)
    db = np.sum(dz, axis=0) 
    return dw, db

def affine_forward(x, w, b):
    out = np.dot(x, w) + b
    cache = (x, w, b)
    return out, cache

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    dx = np.dot(dout, w.T)
    dw = np.dot(x.T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db

def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx

def softmax_loss(x, y): # this is a numerically stable version of cross entropy loss (I learnt this form CS231N)
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx