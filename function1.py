import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import sys

#-----------------------------------------------------
def mnist(PATH):
    #mnist_load
    save_file = PATH

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    train_img,train_label,test_img,test_label = dataset

    #pixel normalization
    X_train, X_test = train_img/255, test_img/255

    #transform_OneHot
    T_train = np.eye(10)[list(map(int,train_label))] 
    T_test = np.eye(10)[list(map(int,test_label))]

    return [X_train,T_train,X_test,T_test]

#-----------------------------------------------------
#math 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def loss(y, t):
    minute = 1e-7
    return -np.sum(np.multiply(t, np.log(y+minute)) + np.multiply((1 - t), np.log(1 - y+minute)))

#-----------------------------------------------------
#batch
def batch(img,label,batch_size):
    img_size = img.shape[0]
    batch_mask = np.random.choice(img_size,batch_size)

    X_batch = img[batch_mask]
    T_batch = label[batch_mask]

    return [X_batch,T_batch]

#-----------------------------------------------------
#convolution
def convolute(IMG,F,Bc,padding,stride):
    IMG_size = int(np.sqrt(IMG.shape[1]))
    F_size = int(F.shape[0])
    batch_size = int(IMG.shape[0])
    FM_size = int((np.sqrt(IMG.shape[1])+2*padding-F_size)/stride+1)
    FM = np.empty((0,FM_size**2))

    if (np.sqrt(IMG.shape[1])+2*padding-F_size)%stride != 0:
        print('★ convolute Error: ストライドが割り切れません')
        print('★ f/stride -> {0}/{1}'.format(int(np.sqrt(IMG.shape[1])+2*padding-F_size),stride))
        sys.exit()

    else:
        pass

    for M in range (batch_size):
        img = np.reshape(IMG[M],(IMG_size,IMG_size))
        img_pad = np.zeros((IMG_size+2*padding,IMG_size+2*padding))
        img_pad[padding:padding+IMG_size,padding:padding+IMG_size]=img
        img_storage = []

        for i in range(FM_size):
            for j in range(FM_size):
                pick_img = img_pad[i*stride:i*stride+F_size,j*stride:j*stride+F_size]
                img_storage = np.append(img_storage,np.tensordot(F,pick_img))

        FM_bias = img_storage+Bc #Bias
        FM_relu = np.where(FM_bias<0,0,FM_bias) #ReLU

        FM = np.vstack((FM,FM_relu))
    return FM

def deconvolute(IMG,delta,F,padding,stride):
    IMG_size = int(np.sqrt(IMG.shape[1]))
    dF_size = int(np.sqrt(delta.shape[1]))
    batch_size = int(IMG.shape[0])
    dFM_size = int((IMG_size+2*padding)-stride*(dF_size-1))
    dFM = np.empty((0,dFM_size**2))
    F_size = int(F.shape[0])

    for M in range (batch_size):
        img = np.reshape(IMG[M],(IMG_size,IMG_size))
        img_pad = np.zeros((IMG_size+2*padding,IMG_size+2*padding))
        img_pad[padding:padding+IMG_size,padding:padding+IMG_size]=img

        dF = np.reshape(delta[M],(dF_size,dF_size))
        img_storage = []

        for i in range(dFM_size):
            for j in range(dFM_size):
                pick_img = img_pad[i*stride:i*stride+dF_size,j*stride:j*stride+dF_size]
                img_storage = np.append(img_storage,np.tensordot(dF,pick_img))

        dFM = np.vstack((dFM,img_storage))
        F_deR = np.where(F<0,0,1)
        dFM_deR = dFM*np.reshape(F_deR,(1,F_size**2))

    return dFM_deR

#-----------------------------------------------------
#forward propagation
def affine(input,W,B,f):
    if input.shape[1] != W.shape[0]:
        print('Affine Error: 入力とWの行列の型が合っていません')
        sys.exit()

    else:
        if f == 'softmax':
            return softmax(np.dot(input, W)+B)

        elif f == 'sigmoid':
            return sigmoid(np.dot(input, W)+B)

        elif f == 'none':
            return np.dot(input, W)+B

        else:
            print('Affine Error: 関数が不明です')
            sys.exit()

def error(Y,label):
    return loss(Y,label)/len(Y)

#-----------------------------------------------------
#accuracy
def accuracy(Y,label):
    batch_size = Y.shape[0]
    Y_accuracy = np.argmax(Y, axis=1)
    T_accuracy = np.argmax(label, axis=1)
    return 100*np.sum(Y_accuracy == T_accuracy)/batch_size

#-----------------------------------------------------
#back propagation
def update(p,learning_rate,delta):
    return p - learning_rate*delta

#-----------------------------------------------------
#show graph
def plot_acc(accuracy_save):
    plt.figure()
    plt.title('ACCURACY')
    plt.xlabel("LEARNING NUMBER(EPOCH)")
    plt.ylabel("ACCURACY (%)")
    # plt.xlim(0, 3000)
    # plt.ylim(0, 100)
    plt.grid(True)
    plt.plot(accuracy_save, color='blue')
    # plt.savefig('/Users/tsutsumifutoshishi/Desktop/cnn_test/plot(accuracy)')
    plt.show()

def plot_loss(E_save):
    plt.figure()
    plt.title('LOSS FUNCTION')
    plt.xlabel("LEARNING NUMBER(EPOCH)")
    plt.ylabel("LOSS VALUE")
    # plt.xlim(0, 3000)
    # plt.ylim(0, 100)
    plt.grid(True)
    plt.plot(E_save, color='blue')
    # plt.savefig('/Users/tsutsumifutoshishi/Desktop/cnn_test/plot(error)')
    plt.show()
    
#-----------------------------------------------------
#img confirm
def show(img,label,i):
    img = np.reshape(img[i],(28,28))

    plt.figure()
    plt.imshow(img, cmap='gray_r')
    plt.show()

    print(label[i])

def show_img(IMG,T_batch,i):
    #Error code
    if i <= IMG.shape[0]:
        pass
    else:
        print("★ Error(show_img):バッチ数よりも大きな値を入力しています")
        sys.exit()

    for i in range(i):
        L = int(np.sqrt(IMG.shape[1]))
        img = np.reshape(IMG[i],(L,L))

        plt.figure()
        plt.imshow(img, cmap='gray_r')
        plt.show()

        print(T_batch[i])

def show_progress(i,num):
    if i%(num/10) == 0:
        print('現在{}%'.format(100*i/num))
    else:
        pass

def save(parameter,name,path):
    save_file = path + '/{}.pkl'.format(name)
    with open(save_file, 'wb') as f:
        pickle.dump(parameter, f) 