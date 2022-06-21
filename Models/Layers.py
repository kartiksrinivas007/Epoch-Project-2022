import numpy as np

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

def cross_entropy_and_sigmoid_backward(cache, y):
    x, w, b, a = cache
    dz = a - y
    dw = np.dot(x.T, dz)
    db = np.sum(dz, axis=0) 
    return dw, db