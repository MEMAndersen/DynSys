# -*- coding: utf-8 -*-
"""
Script to illustrate crowd loading pedestrian dynamics analyses 
to UK NA to BS EN 1991-2
"""

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

# Run crowd loading
crowd_analysis = UKNA_BSEN1991_2_crowd(modalsys_obj=my_sys,
                                       mode_index=0,
                                       bridgeClass=bridgeClass)