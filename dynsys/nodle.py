# -*- coding: utf-8 -*-
"""
Provides functions and methods for importing data and results from NODLE, 
COWI UK's large displacement frame analysis software

@author: RIHY
"""

import pandas
import numpy
import xlrd # for reading from Excel
from mesh import Mesh, DispResults
from common import read_block

# -------------- PUBLIC FUCTIONS ------------------

def read_mesh(fname,name=None):
    """
    Defines mesh based on data provided in COO and MEM tabs
    """
    
    if name is None:
        name = fname[:-5] # strip 'xlsx' from end of filename and use as name
    
    # Define new mesh object
    mesh_obj = Mesh(name=name)
    
    # Read in node data from Excel file
    COO_df = read_COO(fname)
    
    # Define nodes and append to mesh
    mesh_obj.define_nodes(df=COO_df)
    del COO_df
    
    # Read in member data from Excel file
    MEM_df = read_MEM(fname)
    
    # Define elements and append to mesh   
    mesh_obj.define_line_elements(df=MEM_df)
    del MEM_df
    
    return mesh_obj
    
    

def read_COO(fname):
    """
    Reads node data from COO tab and returns as Pandas dataframe
    """
    
    _check_is_xlsx(fname)
    
    df = pandas.read_excel(fname, sheet_name='COO',
                           names=['Node', 'X', 'Y', 'Z'],
                           # note: compressed input not supported
                           dtype={'Node':int, 'X':float, 'Y':float, 'Z':float},
                           skiprows=4,
                           comment=':',
                           usecols=list(range(1,5)))
    
    # Check which axis is vertical
    wb = xlrd.open_workbook(fname,on_demand=True)
    ws = wb.sheet_by_name('OPT')
    vertical_direction = ws.cell_value(7,3) # n.b zero-indexed
    
    if vertical_direction.upper() == 'Y':
        
        print("NODLE model is defined with 'Y' vertical\n" + 
              "Input coordinates will be converted such that Y->Z, Z->(-Y)")
        
        df['Z_temp'] = df['Y']
        df['Y'] = - df['Z']
        df['Z'] = df['Z_temp']
        df = df.drop('Z_temp',axis=1)
    
    return df


def read_MEM(fname):
    """
    Reads member data from MEM tab and returns as Pandas dataframe
    """ 
    
    _check_is_xlsx(fname)
    
    df = pandas.read_excel(fname, sheet_name='MEM',
                           names=['Member', 'EndJ', 'EndK', 'Section'],
                           dtype=int,
                           convert_float=True,
                           skiprows=4,
                           comment=':',
                           usecols=list(range(1,5)))
    
    return df


def read_DIS(fname=None,file_obj=None,mesh_obj=None,verbose=True):
    """
    Reads displacement results, as given in DIS section of NODLE results files
    ***
    Required:
        
    Either of the following inputs are required:
    
    * `fname`, string to denote filename of .res file to be read
    
    * `file_obj`, open textstream object (i.e. partially read data file)
    
    * `mesh_obj`, Mesh object to which results relate. If object provided, 
      results will be assigned to mesh objects e.g. nodes / elements
        
    """
#    ***
#    Optional:
#        
#    * `lcases`, list of indices to denote loadcases to be read. If None, 
#      results for all loadcases will be read
#      
#    * `lcase_type`, code to denote type of referencing to use when selecting 
#      loadcases using `lcases` parameter:
#          
#        * 0 : Select by index, i.e. order in results file (1 = first loadcase)
#          
#        * 1 : Select by loadcase ID
#        
#        * 2 : Select by mode index (dynamic results only) (1 = first mode)
        
    
    if file_obj is None:
        
        if fname is None:
            raise ValueError("Either 'fname' or 'iostream' required!")
        
        file_obj = open(fname,mode='rt')  
    
    EOF = False
    nLoadcases = 0
    nModes = 0
    
    while not EOF:
        
        data = read_block(file_obj,'DIS','FOR')
        
        if data is None:
            EOF=True
            
        else:
            
            lcase_type, title, results = _parse_DIS_data(data)
            
            if lcase_type==1:
                nLoadcases += 1
            elif lcase_type==2:
                nModes += 1
                
            # Assign results to mesh objects
            for node_name, results_arr in results.items():
                
                # Get node object
                node_obj = mesh_obj.node_objs[node_name]
                
                # Determine which container to use
                if lcase_type==1:
                    node_obj.lcase_disp.add(results_arr)
                    
                if lcase_type==2:
                    node_obj.modal_disp.add(results_arr)
                        
    if verbose:
        print("Displacement results read from '%s'\n" % fname +
              "# loadcases found:\t%d\n" % nLoadcases + 
              "# modes founds:\t\t%d" % nModes + 
              "\n")


# -------------- PRIVATE FUNCTIONS ----------------

def _check_is_xlsx(fname):
    if not 'xlsx' in fname:
        raise ValueError("Excel-based NODLE input file expected!")
    
def _parse_DIS_data(data):
    """
    Parses nested list `data` into objects used for storage of loadcase and 
    results data
    """
    
    # Get title line
    title_line = data.pop(0)
    
    # Parse and tidy to obtain key data from title line
    title_dict = {}
    
    title_line = title_line[1:]
    Ndetails = len(title_line)
    
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
    
    return lcase_type, title_dict, results_dict
    


