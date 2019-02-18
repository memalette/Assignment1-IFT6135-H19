import pandas as pd
import numpy as np
from urllib import request
import gzip
import pickle
from sklearn.preprocessing import OneHotEncoder
import random
import timeit
import matplotlib.pyplot as plt
import time

class DataPreprocessing:
    def __init__(self):
        self.filename = "mnist.pkl"


    def load(self):
        with open(self.filename,'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


    def validation_split(self, X, y, val_size=10000, preprocess=False):
        if preprocess is True:
            N = X.shape[0]

            indices = np.arange(N)
            np.random.shuffle(indices)
            idx_val = indices[:val_size]
            idx_train = indices[val_size:]

            pd.DataFrame(idx_train).to_csv("idx_train.csv", index=False, header=False)
            pd.DataFrame(idx_val).to_csv("idx_val.csv", index=False, header=False)

            idx_train = idx_train.reshape((idx_train.shape[0],))
            idx_val = idx_val.reshape((idx_val.shape[0],))

        else:
            idx_train = np.array(pd.read_csv("idx_train.csv"))
            idx_val = np.array(pd.read_csv("idx_val.csv"))
            idx_train = idx_train.reshape((idx_train.shape[0],))
            idx_val = idx_val.reshape((idx_val.shape[0],))

        return X[idx_train,:], y[idx_train], X[idx_val,:], y[idx_val]


    def onehot_encode(self, labels):
        encoder = OneHotEncoder(sparse=False)
        return encoder.fit_transform(labels.reshape(len(labels), 1))




class NN:
    def __init__(self, X_train, y_train, hidden_shape1, hidden_shape2, n_epochs, lr, init_method, X_val, y_val, batch_size):
        # data
        self.input   = X_train 
        self.y       = y_train 
        self.X_val   = X_val
        self.y_val   = y_val

        # dimensions and other hyp.
        self.input_shape   = self.input.shape
        self.output_shape  = self.y.shape
        self.hidden_shape1 = hidden_shape1
        self.hidden_shape2 = hidden_shape2
        self.n_epochs      = n_epochs
        self.lr            = lr
        self.batch_size    = batch_size

        self.initialize_weights(init_method)


    def initialize_weights(self, init_method):
        """ Initializes parameters of the NN

        """
    
        # biases
        self.b1 = np.zeros((self.hidden_shape1, 1))
        self.b2 = np.zeros((self.hidden_shape2, 1))
        self.b3 = np.zeros((self.output_shape[1], 1))

        if init_method is "zero":
            # weights
            self.W1 = np.zeros((self.hidden_shape1, self.input_shape[1]))
            self.W2 = np.zeros((self.hidden_shape2, self.hidden_shape1))
            self.W3 = np.zeros((self.output_shape[1], self.hidden_shape2))

        elif init_method is "gaussian":
            # weights
            self.W1 = np.random.randn(
                            self.hidden_shape1, self.input_shape[1])
            self.W2 = np.random.randn(
                            self.hidden_shape2, self.hidden_shape1)
            self.W3 = np.random.randn(
                            self.output_shape[1], self.hidden_shape2)

        elif init_method is "glorot":
            # uniform intervals
            d1 = np.sqrt(6.0 / (self.input_shape[1] + self.hidden_shape1))
            d2 = np.sqrt(6.0 / (self.hidden_shape2 + self.hidden_shape1))
            d3 = np.sqrt(6.0 / (self.hidden_shape2 + self.output_shape[1]))

            # weights
            self.W1 = np.random.uniform(-d1, d1, 
                            (self.hidden_shape1, self.input_shape[1]))
            self.W2 = np.random.uniform(-d2, d2,
                          (self.hidden_shape2, self.hidden_shape1))
            self.W3 = np.random.uniform(-d3, d3,
                          (self.output_shape[1], self.hidden_shape2))

        # compute number of parameters
        n_param = self.W1.shape[0] * self.W1.shape[1] + self.W2.shape[0] * self.W2.shape[1] + self.W2.shape[0] * self.W2.shape[1] + self.input_shape[1] + self.hidden_shape2 + self.output_shape[1]

        print("[Number of parameters] "+str(n_param))


    def update(self, y, x):
        """ Update weights and biases

        """
        dL_W1, dL_W2, dL_W3, dL_b1, dL_b2, dL_b3 = self.backward(y, x)

        self.W1 -= self.lr * dL_W1
        self.W2 -= self.lr * dL_W2
        self.W3 -= self.lr * dL_W3
        self.b1 -= self.lr * dL_b1
        self.b2 -= self.lr * dL_b2
        self.b3 -= self.lr * dL_b3
        

    def train(self, print_fig=False, fig_name='init'):
        """ Trains the NN

        """
        N = self.input_shape[0]
        
        if np.mod(N, self.batch_size) == 0:
           k = int(N / self.batch_size)
           batch_list = np.ones((k,1), dtype = np.int) * self.batch_size
        else:
           k = int((N-np.mod(N,self.batch_size))/self.batch_size)
           batch_list = np.ones((k+1,1), dtype = np.int) * self.batch_size
           batch_list[-1,:] = np.mod(N, self.batch_size)

        batch_list = list(batch_list.flatten())
        
        losses = {}
        accuracies = {}
        losses['train'], losses['val'] = list(), list()
        accuracies['train'], accuracies['val'] = list(), list()

        for epoch in range(self.n_epochs):
            # shuffle 
            indices = np.arange(N)
            np.random.shuffle(indices)
            x = self.input[indices,:,:]
            y = self.y[indices,:,:]
            
            k = 0
            for batch in batch_list:

                # set batch
                x_batch = x[k:(k+batch),:,:] 
                y_batch = y[k:(k+batch),:,:]

                # feedforward and update parameters
                self.forward(x_batch)
                self.update(y_batch, x_batch)

                k += batch
                print(self.dL_W1)

            # forward entire dataset and and evaluate loss
            # Training set
            l_train, acc_train = self.eval(self.y, self.input)
            losses['train'].append(l_train)
            accuracies['train'].append(acc_train)

            # validation set
            l_val, acc_val = self.eval(self.y_val, self.X_val)
            losses['val'].append(l_val)
            accuracies['val'].append(acc_val)

            print("[epoch] "+str(epoch)+"/"+str(self.n_epochs)+" -- [loss][train/val] "+str(round(l_train,2))+"/"+str(round(l_val,2))+" -- [accuracy][train/val] "+str(round(acc_train,2))+"/"+str(round(acc_val,2)))

        self.last_acc_val = acc_val
        self.last_loss_val = l_val
        self.last_acc_train = acc_train
        self.last_loss_train = l_train

        # keep losses and training time 
        self.losses = losses
        self.accuracies = accuracies


        if print_fig:
            # Loss
            plt.figure(figsize=(10, 10))
            plt.plot(list(range(self.n_epochs)), losses['train'], linewidth=2, label="train")
            plt.plot(list(range(self.n_epochs)), losses['val'], linewidth=2, label="validation")
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('Loss by epoch')
            plt.legend(loc="lower right")
            plt.savefig(fig_name+'_loss.png')
            plt.show()

            # Accuracy
            plt.figure(figsize=(10, 10))
            plt.plot(list(range(self.n_epochs)), accuracies['train'], linewidth=2, label="train")
            plt.plot(list(range(self.n_epochs)), accuracies['val'], linewidth=2, label="validation")
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('Accuracy by epoch')
            plt.legend(loc="lower right")
            plt.savefig(fig_name+'_acc.png')
            plt.show()

    

    def backward(self, y, x):

        # W3
        dL_u3 = self.x3_out-y
        dL_W3 = np.mean(np.transpose(self.x2_out,(0, 2, 1)) * dL_u3,axis=0)
        dL_b3 = np.mean(dL_u3,axis=0)

        # W2
        Iu2 = self.u2_out.copy()
        Iu2[Iu2 < 0] = 0
        Iu2[Iu2 > 0] = 1
        dL_u2 = np.matmul(np.transpose((np.transpose(Iu2,(0, 2, 1)) * self.W3),(0, 2, 1)), dL_u3)
        dL_W2 = np.mean(np.transpose(self.x1_out,(0, 2, 1)) * dL_u2,axis=0)
        dL_b2 = np.mean(dL_u2,axis=0)

        # W1
        Iu1 = self.u1_out.copy()
        Iu1[Iu1 < 0] = 0
        Iu1[Iu1 > 0] = 1
        dL_u1 = np.matmul(np.transpose((np.transpose(Iu1,(0, 2, 1)) * self.W2),(0, 2, 1)), dL_u2)
        
        self.dL_W1 = np.mean(np.transpose(x,(0, 2, 1)) * dL_u1,axis=0)
        dL_b1 = np.mean(dL_u1,axis=0)

        # print(self.x3_out)
        
        return self.dL_W1, dL_W2, dL_W3, dL_b1, dL_b2, dL_b3


    def forward(self, x):
        """ Propagates x through the NN

        """
        # layer 1
        self.u1_out = np.matmul(self.W1, x) + self.b1
        self.x1_out = self.activation(self.u1_out)
        
        # layer 2
        self.u2_out = np.matmul(self.W2, self.x1_out) + self.b2
        self.x2_out = self.activation(self.u2_out)

        # output layer
        self.u3_out = np.matmul(self.W3, self.x2_out) + self.b3
        self.x3_out = self.softmax(self.u3_out)
        

    def activation(self, vector):
        """ Relu activation
        """
        vector[vector < 0] = 0
        return vector


    def eval(self,y,x):
        """ Propagates x through the NN, returns avg loss and accuracy

        """
        self.forward(x)
        L = np.log(self.x3_out) * y
        loss = -np.mean(L.sum(1))

        accuracy = (np.argmax(self.x3_out, axis=1) == np.argmax(y, axis=1)).sum()
        return loss, accuracy/y.shape[0]


    def loss(self, y, x_out):
        """ Computes the cross entropy loss on vector y and x_out

        """
        return -np.dot(y, np.log(x_out))


    def softmax(self, vector):
        """
        softmax matrix
        """
        exps = np.exp(vector - np.max(vector,1).reshape(vector.shape[0],1,1)) 
        return exps / np.sum(exps,1).reshape(vector.shape[0],1,1) 


    def plot_true_vs_finite_diff(self):
        """ Plots the gradients true vs finite difference

        """
        N = [10**0, 5*10**0,10**1,5*10**1,10**2,5*10**2] 
        N.sort()
        
        differences = list()
        for n in N:
            differences.append(self.finite_diff(n))
        
        self.dif=differences.copy()
        plt.figure(figsize=(10, 10))
        fig=plt.plot( N, (differences), linewidth=2)
        plt.xlabel('N')
        plt.ylabel('difference')
        plt.title('Maximum Difference between True and Finite Difference Gradients')
        plt.legend(loc="lower right")
        font = {'family' : 'normal',
        'size'   : 15}
        plt.rc('font', **font)
        plt.show()
        fig.savefig('finite_diff.pdf')
        plt.close(fig)  

    def finite_diff(self, N):
        """ Computes the max between the true and finite difference gradients

        """
            
        p = min(10, self.hidden_shape1 * self.hidden_shape2) 
        W2 = self.W2.copy()
        difference = np.zeros((p,1))
        epsilon = 1 / N
            
        self.forward(self.input[0,:,:].reshape(1,784,1))
        self.backward(self.y[0,:,:].reshape(1,10,1), self.input[0,:,:].reshape(1,784,1))
        for i in range(p):
            self.W2[i,0] = W2[i,0] + epsilon

            L1 = self.eval(self.y[0,:,:].reshape(1,10,1), self.input[0,:,:].reshape(1,784,1))
                
            self.W2[i,0] = W2[i,0] - epsilon
                
            L2 = self.eval(self.y[0,:,:].reshape(1,10,1), self.input[0,:,:].reshape(1,784,1))
                
            difference[i,0] = self.dL_W2[i,0] - (L1-L2)/(2*epsilon)
            self.W2 = W2.copy()
                
            
        print(max(abs(difference)))
        return max(abs(difference))
        


if __name__ == '__main__':

    dp = DataPreprocessing()

    # load MNIST data
    X_trv, y_trv, X_test, y_test = dp.load()

    # 2nd data split 
    X_train, y_train, X_val, y_val = dp.validation_split(X_trv, y_trv, val_size=10000, preprocess=True)
    
    # scale between 0 and 1
    max_mnist = max(X_trv.max(), X_test.max())
    X_train, X_val, X_test = X_train/max_mnist, X_val/max_mnist, X_test/max_mnist

    # One-hot encode the output data
    y_train = dp.onehot_encode(y_train)
    y_test = dp.onehot_encode(y_test)
    y_val = dp.onehot_encode(y_val)

    # crop
    X_train = X_train[0:1000,:]
    X_val   = X_val[0:1000,:]
    X_test  = X_test[0:1000,:]

    y_train = y_train[0:1000,:]
    y_val   = y_val[0:1000,:]
    y_test  = y_test[0:1000,:]

    # data dimensions
    train_size  = X_train.shape[0]
    val_size    = X_val.shape[0]
    test_size   = X_test.shape[0]
    input_size  = X_train.shape[1]
    output_size = y_train.shape[1]
 
    print("ZERO INIT -------------------------------")
    mlp = NN(X_train.reshape(train_size,input_size,1), 
              y_train.reshape(train_size,output_size,1), 
              hidden_shape1 = 400,
              hidden_shape2 = 600, 
              n_epochs = 10, 
              lr = 0.1, 
              init_method = "zero", 
              X_val=X_val.reshape(val_size,input_size,1), 
              y_val=y_val.reshape(val_size,output_size,1),
              batch_size = 16)
    mlp.train(print_fig=True, fig_name='zero')
    train_l_zero = mlp.losses['train']

    print("GAUSSIAN INIT -------------------------------")
    mlp = NN(X_train.reshape(train_size,input_size,1), 
              y_train.reshape(train_size,output_size,1), 
              hidden_shape1 = 400,
              hidden_shape2 = 600, 
              n_epochs = 10, 
              lr = 0.01, 
              init_method = "gaussian", 
              X_val=X_val.reshape(val_size,input_size,1), 
              y_val=y_val.reshape(val_size,output_size,1),
              batch_size = 16)
    mlp.train(print_fig=True, fig_name='gaussian')
    train_l_gaussian = mlp.losses['train']

    print("GLOROT INIT -------------------------------")
    mlp = NN(X_train.reshape(train_size,input_size,1), 
              y_train.reshape(train_size,output_size,1), 
              hidden_shape1 = 400,
              hidden_shape2 = 600, 
              n_epochs = 10, 
              lr = 0.1, 
              init_method = "glorot", 
              X_val=X_val.reshape(val_size,input_size,1), 
              y_val=y_val.reshape(val_size,output_size,1),
              batch_size = 16)
    mlp.train(print_fig=True, fig_name='glorot')
    train_l_glorot = mlp.losses['train']

    # save results
    df = pd.DataFrame(data={"loss_glorot": train_l_glorot, "loss_zero": train_l_zero, "loss_gaussian": train_l_gaussian})
    df.to_csv("./init_results.csv", sep=',',index=False)

    # grid search 
    grid_size      = 10000
    max_param      = 1000000
    min_param      = 500000
    in_size        = 784
    out_size       = 10
    learning_rates = [1e-1, 1e-2, 1e-3]
    batch_size     = [16, 32, 64, 128]
    hidden1        = [200, 400, 600, 800]
    hidden2        = [200, 400, 600, 800]

    combinations = list()
    i = 1
    for lr in learning_rates:
        for bs in batch_size:
            for h1 in hidden1:
                for h2 in hidden2:
                    n_param = in_size*h1 + h1*h2 + out_size*h2 + h2 + in_size + out_size
                    if (n_param < max_param):
                        if (n_param > min_param):
                            combinations.append([lr,bs,h1,h2])
                            print([lr,bs,h1,h2])
                            i = i + 1
    print(i-1)


    acc_val = list()
    acc_train = list()

    i = 1
    for comb in combinations:
        print("[Combination "+str(i)+"] -- "+str(comb))
        lr, batch_size, hidden_shape1, hidden_shape2 = comb[0], comb[1], comb[2], comb[3]

        indices = np.arange(grid_size)
        np.random.shuffle(indices)

        mlp = NN(X_train[indices,:].reshape(grid_size,input_size,1), 
                  y_train[indices,:].reshape(grid_size,output_size,1), 
                  hidden_shape1=hidden_shape1,
                  hidden_shape2=hidden_shape2, 
                  n_epochs=10, 
                  lr=lr, 
                  init_method="glorot", 
                  X_val=X_val[indices,:].reshape(grid_size,input_size,1), 
                  y_val=y_val[indices,:].reshape(grid_size,output_size,1),
                  batch_size=batch_size)

        mlp.train()
        acc_val.append(mlp.last_acc_val)
        acc_train.append(mlp.last_acc_train)


        # save results
        df = pd.DataFrame(data={"accuracy_train": acc_train, "accuracy_val": acc_val})

        df.to_csv("./results.csv", sep=',',index=True)

        i += 1
