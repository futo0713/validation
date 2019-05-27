import function1
import numpy as np
import time

#minist_load
save_file = '/Users/tsutsumifutoshishi/Desktop/MNIST_test/mnist.pkl'
X_train,T_train,X_test,T_test = function1.mnist(save_file)

#initial settings
IMG_size =28

F_size = 5
F = np.random.randn(F_size,F_size)

FM_size = int(np.sqrt(X_train.shape[1])-F_size+1)

Bc = np.random.randn(1,FM_size**2)

Wo = np.random.randn(FM_size**2, 10)
Bo = np.random.randn(1,10)

learning_rate = 0.001

E_save = []
accuracy_save = []
start_time = time.time()

batch_size = 100

#iteration
num_of_itr=10000
for i in range(num_of_itr):
    function1.show_progress(i,num_of_itr)

    X_batch,T_batch = function1.batch(X_train,T_train,batch_size)

    FM_batch = np.empty((0,FM_size**2))
    FM_bias_batch = np.empty((0,FM_size**2))
    for M in range (batch_size):
        img = np.reshape(X_batch[M],(IMG_size,IMG_size))
        FM_storage = []

        for i in range(FM_size):
            for j in range(FM_size):
                pick_img = img[i:i+F_size, j:j+F_size]
                FM_storage = np.append(FM_storage,np.tensordot(F,pick_img))

        FM_bias = FM_storage+Bc #Bias
        FM_relu = np.where(FM_bias<0,0,FM_bias) #ReLU

        FM_batch = np.vstack((FM_batch,FM_relu))
        FM_bias_batch = np.vstack((FM_bias_batch,FM_bias))

    Y = function1.affine(FM_batch,Wo,Bo,'softmax')

    E = function1.error(Y,T_batch)
    E_save = np.append(E_save, E)

    Acc = function1.accuracy(Y,T_batch)
    accuracy_save = np.append(accuracy_save, Acc)

    #======================================================
    dWo = np.dot(FM_batch.T,(Y-T_batch))
    dBo = np.reshape(np.sum(Y-T_batch, axis=0),(1,10))

    #------------------------------------------------------
    delta = np.where(FM_bias_batch<=0,0,1)*np.dot(Y-T_batch,Wo.T)

    dF_batch = np.empty((0,F_size**2))
    for M in range (batch_size):
        img = np.reshape(X_batch[M],(IMG_size,IMG_size))
        img_delta = np.reshape(delta[M],(FM_size,FM_size))
        dF_storage = []

        for i in range(F_size):
            for j in range(F_size):
                pick_img = img[i:i+FM_size, j:j+FM_size]
                dF_storage = np.append(dF_storage,np.tensordot(img_delta,pick_img))

        dF_batch = np.vstack((dF_batch,dF_storage))

    dF = np.reshape(np.average(dF_batch,axis=0),(F_size,F_size))

    dBc = np.reshape(np.average(delta,axis=0),(1,FM_size**2))

    #======================================================
    Wo = function1.update(Wo,learning_rate,dWo)
    Bo = function1.update(Bo,learning_rate,dBo)
    F = function1.update(F,learning_rate,dF)
    Bc = function1.update(Bc,learning_rate,dBc)

end_time = time.time()
total_time = end_time - start_time
print(total_time)

#show graph
function1.plot_acc(accuracy_save)
function1.plot_loss(E_save)

#パラメータ保存
#-----------------------------------------------------
CNN_parameters = [Wo,Bo,F,Bc]
path = '/Users/tsutsumifutoshishi/Desktop/cnn_test'
name = '/CNN_ver6'

function1.save(CNN_parameters,name,path)