# -*- coding: utf-8 -*-
"""
Provides functions and methods for importing data and results from NODLE, 
COWI UK's large displacement frame analysis software

@author: RIHY
"""

import pandas
from mesh import Mesh

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
    mesh_obj.define_elements(df=MEM_df)
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


def read_DIS(fname=None,iostream=None,lcases=None,lcase_type=1):
    """
    Reads displacement results, as given in DIS section of NODLE results files
    ***
    Required:
        
    Either of the following inputs are required:
    
    * `fname`, string to denote filename of .res file to be read
    
    * `iostream`, open filestream object (i.e. partially read data file)
    
    ***
    Optional:
        
    * `lcases`, list of indices to denote loadcases to be read. If None, 
      results for all loadcases will be read
      
    * `lcase_type`, code to denote type of referencing to use when selecting 
      loadcases using `lcases` parameter:
          
        * 0 : Select by index, i.e. order in results file (1 = first loadcase)
          
        * 1 : Select by loadcase ID
        
        * 2 : Select by mode index (dynamic results only) (1 = first mode)
        
    """
    
    if iostream is None:
        
        if fname is None:
            raise ValueError("Either 'fname' or 'iostream' required!")
            
        pass
    


# -------------- PRIVATE FUNCTIONS ----------------

def _check_is_xlsx(fname):
    if not 'xlsx' in fname:
        raise ValueError("Excel-based NODLE input file expected!")
    


