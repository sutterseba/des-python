import DES
import numpy as np

def new_key (custom_seed = None):

    if custom_seed:
        np.random.seed(custom_seed)
    
    return np.random.randint(2, size=192)

def encrypt (x, k):

    # Split 3DES key into individual DES keys
    k = np.split(k, 3)

    return DES.encrypt(DES.encrypt(DES.encrypt(x, k[0]), k[1]), k[2])

def decrypt (y, k):

    # Split 3DES key into individual DES keys
    k = np.split(k, 3)

    return DES.decrypt(DES.decrypt(DES.decrypt(y, k[2]), k[1]), k[0])