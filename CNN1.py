import function1
import numpy as np
import time

#minist_load
save_file = '/Users/tsutsumifutoshishi/Desktop/MNIST_test/mnist.pkl'
X_train,T_train,X_test,T_test = function1.mnist(save_file)

# function1.show(X_train,T_train,60)

#-----------------------------------------------------
#initial settings
F_size = 5
F = np.random.randn(F_size,F_size)

Wo = np.random.randn(576, 10)
Bo = np.random.randn(1,10)
learning_rate = 0.001

E_save = []
accuracy_save = []
start_time = time.time()

batch_size = 100
#-----------------------------------------------------
#iteration
num_of_itr=1
for i in range(num_of_itr):
    #batch
    X_batch,T_batch = function1.batch(X_train,T_train,batch_size)

    #convolute
    C = function1.convolute(X_batch,F)
    Y = function1.affine(C,Wo,Bo,'softmax')

    E = function1.error(Y,T_batch)
    E_save = np.append(E_save, E)

    Acc = function1.accuracy(Y,T_batch)
    accuracy_save = np.append(accuracy_save, Acc)
    print(Y.shape)