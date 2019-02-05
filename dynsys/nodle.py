# -*- coding: utf-8 -*-
"""
Provides functions and methods for importing data and results from NODLE, 
COWI UK's large displacement frame analysis software

@author: RIHY
"""

import pandas

def read_COO(fname):
    """
    Reads data from COO tab and returns as Pandas dataframe
    """
    
    df = pandas.read_excel(fname, sheet_name='COO',
                           names=['Node', 'X', 'Y', 'Z'],
                           index_col=0, # use Node ID as index
                           # note: compressed input not supported
                           dtype={'Node':int, 'X':float, 'Y':float, 'Z':float},
                           skiprows=4,
                           comment=':',
                           usecols=list(range(1,5)))
    
    return df

def read_MEM(fname):
    """
    Reads data from MEM tab and returns as Pandas dataframe
    """    
    
    df = pandas.read_excel(fname, sheet_name='MEM',
                           names=['Member', 'EndJ', 'EndK', 'Section'],
                           index_col=0, # use Member ID as index
                           # note: compressed input not supported
                           convert_float=True,
                           skiprows=4,
                           comment=':',
                           usecols=list(range(1,5)))
    
    return df




