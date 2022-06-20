import numpy as np


class Solver():
    def __init__(self, model, X_train, y_train, lr = 0.05, batch_size = 20, num_epochs = 10, print_every = 1000):
        self.lr = lr
        self.data = {}
        self.data['X_train'] = X_train
        self.data['y_train'] = y_train
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.loss_history = np.array([])
        self.grad_history = np.array([])
        pass
    def train(self):
        for i in range(self.num_epochs):
            for j in range(self.data['X_train'].shape[0] // self.batch_size):
                X_batch = self.data['X_train'][j * self.batch_size:(j + 1) * self.batch_size, :]
                y_batch = self.data['y_train'][j * self.batch_size:(j + 1) * self.batch_size].reshape(-1,1)
                loss, grads = self.model.loss(X_batch, y_batch)
                self.lr = self.lr
                self.model.params['W'] = self.model.params['W'] - self.lr * grads['W']
                self.model.params['b'] = self.model.params['b'] - self.lr * grads['b']
                if(j % self.print_every == 0):
                    print("Epoch = ", i, "Batch = ", j, "Loss = ", loss, "Gradient_max = ", np.max(abs(grads['W'])), "learning rate = ", self.lr)
                    self.loss_history = np.append(self.loss_history, loss)
                    self.grad_history = np.append(self.grad_history, np.sum(grads['W'] * grads['W']))
        