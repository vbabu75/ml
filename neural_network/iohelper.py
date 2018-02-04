import scipy.io as sio
import numpy as np

def get_var_names(fn):
    """
    This function returns back the names of variables defined in the given
    .mat file
    """
    temp = sio.loadmat(fn)
    return [key for key in temp.keys() if not key.startswith('_')]

def save_as_npy(fn):
    """
    This function opens the given .mat file and saves all the arrays
    defined inside it as individual .npy file.
    """
    var_names = get_var_names(fn)
    temp = sio.loadmat(fn)
    for name in var_names:
        np.save(name+".npy",temp[name])

