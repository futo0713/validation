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

padding = 1
stride = 1

FM_size = int((np.sqrt(X_train.shape[1])+2*padding-F_size)/stride+1)

Bc = np.random.randn(1,FM_size**2)

Wo = np.random.randn(FM_size**2, 10)
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
    function1.show_progress(i,num_of_itr)
    X_batch,T_batch = function1.batch(X_train,T_train,batch_size)
    # function1.show_img(X_batch,T_batch,1)

    #=========================================
    C = function1.convolute(X_batch,F,Bc,padding,stride)
    # function1.show_img(C,T_batch,1)

    Y = function1.affine(C,Wo,Bo,'softmax')

    E = function1.error(Y,T_batch)
    E_save = np.append(E_save, E)

    Acc = function1.accuracy(Y,T_batch)
    accuracy_save = np.append(accuracy_save, Acc)

    #=========================================
    dWo = np.dot(C.T,(Y-T_batch))
    dBo = np.reshape(np.sum(Y-T_batch, axis=0),(1,10))

    delta = np.dot(Y-T_batch,Wo.T)
    print(delta.shape)
    dF = function1.deconvolute(X_batch,delta) 
    dF = np.average(dF, axis=0)
    print(dF.shape)
#     dF = np.reshape(dF,(5,5)) 

#     dBc = delta
#     #=========================================
#     Wo = function1.update(Wo,learning_rate,dWo)
#     Bo = function1.update(Bo,learning_rate,dBo)
#     F = function1.update(F,learning_rate,dF)
#     Bc = function1.update(Bc,learning_rate,dBc)

# end_time = time.time()
# total_time = end_time - start_time
# print(total_time)

# #show graph
# function1.plot_acc(accuracy_save)
# function1.plot_loss(E_save)