# -*- coding: utf-8 -*-
"""
Common functions used by other modules
"""

# -------- TYPE CONVERSION / CHECKER METHODS ---------------

import numpy as npy
import inspect

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


def convert2matrix(value,dtype=float):
    """
    Checks `value` is of numpy matrix format; 
    attempts to convert to matrix if not
    """
    
    if not isinstance(value,npy.matrix):
        
        value = npy.array(value)
        
        if value.ndim <= 2:
            value = npy.asmatrix(value,dtype=dtype)
        
        else:
            raise ValueError("Error: numpy.matrix type expected. " + 
                             "Could not convert")
            
    return value



# Checking functions
def check_is_class(obj):
    if not inspect.isclass(obj):
        raise ValueError("Class expected!\n" + 
                         "type(obj): {0}".format(type(obj)))
        
        
def check_class(obj,expected_class:str):
    """
    Returns exception if supplied object is not of the expected class
    """
    
    if not isinstance(obj, expected_class):
        
        raise ValueError("Unexpected object provided!\n" + 
                         "provided:\t{0}\n".format(type(obj)) + 
                         "expected:\t{0}".format(expected_class))
                         
    
def check_shape(value,expected_shape:tuple):
    
    if value.shape != expected_shape:
        raise ValueError("Unexpected shape!")
        

def check_is_square(value):
        
    if value.ndim != 2:
        raise ValueError("value.ndim != 2")
        
    if value.shape[0] != value.shape[1]:
        raise ValueError("Non-square matrix provided!")

# ----------- DEPRECATION -----------------

def deprecation_warning(old_method:str,new_method):
    
    print("*** Warning: '%s' method is deprecated. Use '%s' instead ***" % 
          (old_method,new_method))


# -------- FILE READING FUNCTIONS ----------

def is_within_string(full_str,str2find):
    """
    Tests whether a given string, `str2find`, can be found within a different 
    string, `full_str`. Returns True if string is found, false otherwise
    """
    index = full_str.find(str2find)
    if index == -1:
        return False
    else:
        return True
    
    
def read_line_conditional(f,func,*fargs):
    """
    Function to implement 'conditional peeking' when working with files
    
    Function works as follows:
    
    * Next line is read from filestream `f`.
    
    * String data, as read-in, is tested by supplied function `func`. The 
      first argument of `func` should be the next line string, as read from the 
      file. Other arguments may be supplied via `fargs` list
      
    * If `func` returns `True` then revert to previous line, i.e. do not 
      advance through file. Otherwise advance through file as usual when 
      reading line-by-line
    """
    
    pos = f.tell()
    line = f.readline()
    
    if len(line) == 0:
        # denotes end of file
        return -1
    
    revert = func(line,*fargs)
    if revert:
        f.seek(pos)
        return None
    else:
        return line
    
    
def read_block(f, start_str, end_str):
    """
    Function to read a block of comma-delimited data from a file
    
    Data is read within a block bounded by `start_str` and `end_str`, inclusive 
    and exclusive respectively.
    """
    
    data=[]
    
    # Locate start of block within file, as denoted by start_str
    start_found = False
    while not start_found:
        
        line = read_line_conditional(f,is_within_string,'DIS')
    
        if line is None:
            start_found=True
            
        if line == -1:
            # EOF detected
            return None # break from function
        
    # Locate end pf block
    end_found = False
    while not end_found:
        
        line = read_line_conditional(f,is_within_string,'FOR')
        if line is None:
            end_found=True
        else:
            
            line = line.replace('/\n','')
            data_vals = line.split(',')
            data.append(data_vals)
            
    return data

# ------------ PLOTTING --------------
    
def set_equal_aspect_3d(ax,axis_limits):
    """
    Draws a hidden diagonal line to ensure square axes obtained when plotting 
    in 3d
    
    _This is a workaround relating to known matplotlib shortcoming. May not 
    be required once proper API functionality provided_
    """
    
    centre = npy.mean(axis_limits,axis=1)
    sz = npy.max([a[1]-a[0] for a in axis_limits])
    
    a = centre + 0.5*sz*npy.ones((3,))
    b = centre - 0.5*sz*npy.ones((3,))
    ab = npy.vstack((a,b))
    
    ax.plot(ab[:,0],ab[:,1],ab[:,2],'k',linestyle='None') # hidden line