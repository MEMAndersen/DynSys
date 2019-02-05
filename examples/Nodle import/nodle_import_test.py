# -*- coding: utf-8 -*-
"""
Script to test and demonstrate methods for importing data from NODLE excel 
template

@author: RIHY
"""

import nodle

fname = 'NODLE_demo.xlsx'

COO_df = nodle.read_COO(fname)
print(COO_df)

MEM_df = nodle.read_MEM(fname)
print(MEM_df)

