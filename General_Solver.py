import numpy as np

class GeneralSolver():
    def __init__ (self,model, X_train, y_train, lr = 0.05, batch_size = 20,num_epochs = 10 , print_every = 1000, mu = 0.95):
        self.lr = lr
        self.data = {}
        self.data['X_train'] = X_train
        self.data['y_train'] = y_train
        self.model = model
        self.velocity = {}
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.loss_history = np.array([])
        self.grad_history = np.array([])
        self.vel_history = np.array([])
        self.loss_final_history = np.array([])
        self.grad_final_history = np.array([])       
        self.mu = mu
        pass
    def train(self, mode):
        for param, value in self.model.params.items():
            self.velocity[param] = np.zeros_like(value)
        
        print('fine till here !')

        for i in range(self.num_epochs):
            for j in range(self.data['X_train'].shape[0] // self.batch_size):
                X_batch = self.data['X_train'][j * self.batch_size:(j + 1) * self.batch_size, :]
                y_batch = self.data['y_train'][j * self.batch_size:(j + 1) * self.batch_size].reshape(-1,1)
                loss, grads = self.model.loss(X_batch, y_batch)
                for param,value in self.model.params.items():
                    if mode == 'sgd':
                        self.sgd_update(param,grads[param])
                    elif mode == 'sgd_momentum':
                        self.momentum_update(param,grads[param])
                        
                if(j  == 0):
                    print("Epoch = ", i, "Batch = ", j, "Loss = ", loss)
                    self.loss_history = np.append(self.loss_history, loss)
                    # self.grad_history = np.append(self.grad_history, np.linalg.norm(grads['W1']))            
                    # self.loss_history = np.append(self.loss_history, loss)
                    # self.grad_history = np.append(self.grad_history, np.sum(grads['W1'] * grads['W1']))
                    # self.vel_history = np.append(self.vel_history, np.sum(v_w * v_w))
                if(j == (self.data['X_train'].shape[0] // self.batch_size - 1)):
                    self.loss_final_history = np.append(self.loss_final_history, loss)
                    # self.grad_2_history = np.append(self.grad_2_history, np.linalg.norm(grads['W1']))
    
    def sgd_update(self, param , grad):
        self.model.params[param] +=  -1 * self.lr * grad
    

    def momentum_update(self, param , grad):
        self.velocity[param] = self.mu * self.velocity[param] - self.lr * grad
        self.model.params[param] += self.velocity[param]
        


        


        