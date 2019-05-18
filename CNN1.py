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

#-----------------------------------------------------
#iteration
num_of_itr=1
for i in range(num_of_itr):
    print(F)