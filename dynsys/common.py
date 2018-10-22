# -*- coding: utf-8 -*-
"""
Common functions used by other modules
"""

"""
Type conversion / checker methods
"""

import numpy as npy

# Type conversion / checking functions
# Common usage: use as decorators to setter functions
    
def convert2array(value):
    """
    Checks `value` is of numpy array format; 
    attempts to convert to array if not
    """
    
    if not isinstance(value,npy.ndarray):
        
        value = npy.array(value)
        
    return value


def convert2matrix(value):
    """
    Checks `value` is of numpy matrix format; 
    attempts to convert to matrix if not
    """
    
    if not isinstance(value,npy.matrix):
        
        value = npy.array(value)
        
        if value.ndim == 2:
            value = npy.asmatrix(value,dtype=complex)
        
        else:
            raise ValueError("Error: numpy.matrix type expected. " + 
                             "Could not convert")
            
    return value



# Shape checking functions
    
def check_shape(value,expected_shape:tuple):
    
    if value.shape != expected_shape:
        raise ValueError("Unexpected shape!")
        

def check_is_square(value):
        
    if value.ndim != 2:
        raise ValueError("value.ndim != 2")
        
    if value.shape[0] != value.shape[1]:
        raise ValueError("Non-square matrix provided!")



