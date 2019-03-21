# -*- coding: utf-8 -*-
"""
Example script to demonstrate how a set of TMDs can be appended to a 
modal system with multiple modes

"""

# Std python imports
import numpy as npy
from copy import deepcopy
import matplotlib.pyplot as plt

# DynSys imports
import modalsys
from damper import TMD

plt.close('all')

#%%

# Define modal system, based on input files
modal_params={}
modal_params['Mass'] = 362000
modal_params['Freq'] = 0.89
modal_params['DampingRatio'] = 0.007
modal_sys = modalsys.ModalSys(name='Main system',
                              modalParams_dict=modal_params)
modal_sys.add_outputs()

#%%

# Define TMDs
TMD1 = TMD(name='TMD-1',sprung_mass=5000, nat_freq=0.88, damping_ratio=0.15)
TMD2 = TMD(name='TMD-2',sprung_mass=4800, nat_freq=0.92, damping_ratio=0.10)

#%%

# Append TMDs to modal system
modal_sys.append_system(TMD1,Xpos_parent=50.0,DOF_child=0)
modal_sys.append_system(TMD2,Xpos_parent=60.0,DOF_child=0)

modal_sys.plot_modeshapes()

#%%

# Get eigenvalues of system with TMDs
eig_rslts = modal_sys.calc_eigenproperties()
eig_rslts.plot()

#%%

# Get frequency response of systems with and without TMDs
freq_response_rslts = modal_sys.calc_freq_response(fmax=2.0)
freq_response_rslts.plot()

