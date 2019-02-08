# -*- coding: utf-8 -*-
"""
Script to test and demonstrate methods for importing data from NODLE excel 
template

@author: RIHY
"""

import nodle
from common import read_block
import matplotlib.pyplot as plt
plt.close('all')


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
results_fname = 'NODLE_demo.res'
nodle.read_DIS(results_fname,mesh_obj=mesh_obj)

#%%
# Spot-check results have been assigned to nodes properly
nodeID = 103
node_obj = mesh_obj.node_objs[nodeID]
print(node_obj.lcase_disp)
print(node_obj.modal_disp)

#%%
mesh_obj.plot_results(results_type='disp',lcase=1,lcase_type=2)
