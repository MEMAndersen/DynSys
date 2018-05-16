# -*- coding: utf-8 -*-
"""
Script to illustrate crowd loading pedestrian dynamics analyses 
to UK NA to BS EN 1991-2
"""

# Python dist imports
import numpy
import matplotlib.pyplot as plt

# DynSys package imports
import modalsys
from dyn_analysis import UKNA_BSEN1991_2_crowd

#%%

bridgeClass='b'

# Define modal system
my_sys = modalsys.ModalSys(name="Bridge example",fname_modeshapes=None)

# Add output matrix
my_sys.AddOutputMtrx()

#%%

# Define files to define modeshapes along edges of deck strips
mshape_fnames = [["modeshapes_edge_MS_NSS.csv","modeshapes_CL_MS_NSS.csv"],
                 ["modeshapes_CL_MS_SSS.csv","modeshapes_edge_MS_SSS.csv"],
                 ["modeshapes_CL_NSS_SSS.csv","modeshapes_edge_NSS_SSS.csv"]]
mshape_fnames = numpy.array(mshape_fnames)

# Define functions to define how wdith varies with chainage in each region

def width_region1(x):
    return 1.5+0.01*x

width_funcs = [1.5,width_region1,2.0]

# Run crowd loading
crowd_analysis = UKNA_BSEN1991_2_crowd(modalsys_obj=my_sys,
                                       bridgeClass=bridgeClass,
                                       width_func_list=width_funcs,
                                       modeshapes_fname_arr=mshape_fnames)

target_acc = crowd_analysis.run(mode_index=2,load_intensity=2.0)
crowd_analysis.plot_results()

