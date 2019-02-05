# -*- coding: utf-8 -*-
"""
Useful functions which may be used be other modules

@author: RIHY
"""

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

