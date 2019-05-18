import numpy as np
import sys

#-----------------------------------------------------
# Feature Map画像の大きさ
def FM_size(img,F_size):
    return int(np.sqrt(img.shape[1])-F_size+1)

