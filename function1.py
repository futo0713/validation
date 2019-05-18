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

def show(img,label,i):
    img = np.reshape(img[i],(28,28))

    plt.figure()
    plt.imshow(img, cmap='gray_r')
    plt.show()

    print(label[i])

#-----------------------------------------------------
#math 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def loss(y, t):
    delta = 1e-7
    return -np.sum(np.multiply(t, np.log(y+delta)) + np.multiply((1 - t), np.log(1 - y+delta)))

#-----------------------------------------------------
#batch
def batch(img,label,batch_size):
    img_size = img.shape[0]
    batch_mask = np.random.choice(img_size,batch_size)

    X_batch = img[batch_mask]
    T_batch = label[batch_mask]

    return [X_batch,T_batch]

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
    plt.show()