# -*- coding: utf-8 -*-
"""
Example to demonstrate (and test) use of constraints
"""

# dynsys library imports
import modalsys
from damper import TMD
from ped_dyn import UKNA_BSEN1991_2_walkers_joggers

# Define a modal system
my_modal_sys = modalsys.ModalSys(name="my_modal_sys")
my_modal_sys.AddOutputMtrx(fName='outputs.csv')

# Define some TMD systems
TMD1 = TMD(sprung_mass=10,nat_freq=1.0,damping_ratio=0.1)
TMD2 = TMD(sprung_mass=20,nat_freq=1.2,damping_ratio=0.15)

# Append TMDs to modal system
my_modal_sys.AppendSystem(child_sys=TMD1,Xpos_parent=30.0,DOF_child=0)
my_modal_sys.AppendSystem(child_sys=TMD2,Xpos_parent=50.0,DOF_child=0)
my_modal_sys.PrintSystemMatrices(printValues=True)

#%%
# Compute eigenproperties of system with TMDs
eig_results = my_modal_sys.CalcEigenproperties(makePlots=True)

#%%
# Plot modeshapes
my_modal_sys.PlotModeshapes()

#%%
# Carry out pedestrian dynamics analysis
analysis_obj = UKNA_BSEN1991_2_walkers_joggers(modalsys_obj=my_modal_sys,
                                               mode_index=0)

results_obj = analysis_obj.run()

#%%
results_obj.PlotResults(dynsys_obj=my_modal_sys)
results_obj.PlotResponsePSDs()

