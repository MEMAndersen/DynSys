# -*- coding: utf-8 -*-
"""
Script to test and demonstrate methods for importing data from NODLE excel 
template

@author: RIHY
"""

import nodle
import numpy
import matplotlib.pyplot as plt
plt.close('all')

from collections import OrderedDict

#%%

fname = 'NODLE_demo.xlsx'

COO_df = nodle.read_COO(fname)
print(COO_df)

MEM_df = nodle.read_MEM(fname)
print(MEM_df)

mesh_obj = nodle.read_mesh(fname)

mesh_obj.define_gauss_points(N_gp=2)

mesh_obj.plot(plot_gps=True)

#%%
#
#fname = 'bigger_model.xlsx'
#mesh_obj = nodle.read_mesh(fname)
#mesh_obj.define_gauss_points(N_gp=1)
#mesh_obj.plot(plot_gps=True)

#%%

def is_within_string(full_str,str2find):
    index = full_str.find(str2find)
    if index == -1:
        return False
    else:
        return True

def read_line_conditional(f,func,*fargs):
    """
    Function to implement 'conditional peeking' when working with files
    
    * Next line is read from filestream `f`.
    
    * String data as read-in is tested by supplied function `test_func`. The 
      function provided should take a single argument, which is the next line 
      in the file
      
    * If function returns `True` then revert to previous line, i.e. do not 
      advance through file
    """
    
    pos = f.tell()
    line = f.readline()
    revert = func(line,*fargs)
    if revert:
        f.seek(pos)
        return None
    else:
        return line
 
def read_block_from_file(f, start_str, end_str):
    
    data=[]
    
    # Locate start of block within file, as denoted by start_str
    start_found = False
    while not start_found:
        
        line = read_line_conditional(f,is_within_string,'DIS')
    
        if line is None:
            start_found=True
        
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

def parse_DIS_data(data):
    
    # Get title line
    title_line = data.pop(0)
    
    # Parse and tidy to obtain key data from title line
    title_dict = {}
    
    title_line = title_line[1:]
    Ndetails = len(title_line)
    print("Ndetails = %d" % Ndetails)
    
    if Ndetails == 1:
        # Ordinary loadcase results
        lcase_type = 1
        
        loadcase_title = title_line[0]
        loadcase_title = loadcase_title.replace("'","")
        loadcase_title = loadcase_title.lstrip()
        loadcase_details = loadcase_title.split(':')
        
        name, description = loadcase_details
        
        name = int(name)
        
        title_dict['Name'] = name
        title_dict['Description'] = description.lstrip()
        
    elif Ndetails == 3:
        # Modal results
        lcase_type = 2
        
        mode_name, freq, mass = title_line
        
        mode_name = int(mode_name[6:]) # remove 'Mode ' prefix
        freq = float(freq[3:].lstrip()[:-2])
        mass = float(mass[3:].lstrip()[:-1])
        
        title_dict['Mode'] = mode_name
        title_dict['Frequency'] = freq
        title_dict['Mass'] = mass
        
    else:
        raise ValueError("Unexpected title encountered")
    
    # Discard last 3 lines (max/min displacement summary)
    data.pop(-1)
    data.pop(-1)
    data.pop(-1)
    
    # Collate node-displacement data into dict
    results_dict = {}
    
    for row in data:
        
        node = int(row[2])
        disp_vals = numpy.array(row[3:],dtype=float)
        
        results_dict[node]=disp_vals
    
    print(results_dict)
    print(title_dict)
    
    return lcase_type, title_dict, results_dict
    

#%%
results_fname = 'NODLE_demo.res'
file_obj = open(results_fname,mode='rt')    
data = read_block_from_file(file_obj,'DIS','FOR')
lcase_type, title,results = parse_DIS_data(data)
data = read_block_from_file(file_obj,'DIS','FOR')
lcase_type, title,results = parse_DIS_data(data)
data = read_block_from_file(file_obj,'DIS','FOR')
lcase_type, title,results = parse_DIS_data(data)
#nodle.read_DIS(results_fname,mesh_obj)