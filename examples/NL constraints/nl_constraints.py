# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:42:44 2018

@author: whoever
"""

import matplotlib.pyplot as plt

plt.close('all')

# DynSys package imports
import modalsys
from damper import TMD
from ped_dyn import UKNA_BSEN1991_2_walkers_joggers

# Define modal system
my_sys = modalsys.ModalSys(name="Bridge example")

# Define a system to append
TMD1 = TMD(sprung_mass=2000,nat_freq=1.9,damping_ratio=0.1,name='TMD1')

# Statically append TMDs to modal system
my_sys.AppendSystem(child_sys=TMD1,Xpos_parent=58.0,DOF_child=0)

my_sys.PlotModeshapes()

# Define single walkers/joggers analysis
my_analysis = UKNA_BSEN1991_2_walkers_joggers(modalsys_obj=my_sys,
                                              mode_index=1,
                                              analysis_type="joggers")
                                       
results_obj = my_analysis.run()
tstep_obj = results_obj.tstep_obj