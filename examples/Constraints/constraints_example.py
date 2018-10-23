# -*- coding: utf-8 -*-
"""
Example to demonstrate (and test) use of constraints
"""

import matplotlib.pyplot as plt
plt.close('all')

# dynsys library imports
import modalsys
from dynsys import PlotFrequencyResponse
from damper import TMD
from ped_dyn import UKNA_BSEN1991_2_walkers_joggers

#%%

# Define a modal system
my_modal_sys = modalsys.ModalSys(name="my_modal_sys")
my_modal_sys.AddOutputMtrx(fName='accn_outputs.csv')
my_modal_sys.AddOutputMtrx(fName='disp_outputs.csv')

#%%

# Define some TMD systems
TMD1 = TMD(sprung_mass=10,nat_freq=1.0,damping_ratio=0.1,name='TMD1')
TMD2 = TMD(sprung_mass=20,nat_freq=1.2,damping_ratio=0.15,name='TMD2')

#%%

# Append TMDs to modal system
my_modal_sys.AppendSystem(child_sys=TMD1,Xpos_parent=30.0,DOF_child=0)
my_modal_sys.AppendSystem(child_sys=TMD2,Xpos_parent=50.0,DOF_child=0)
my_modal_sys.PrintSystemMatrices(printValues=True)

#%%
# Compute eigenproperties of system with TMDs
eig_results = my_modal_sys.CalcEigenproperties(makePlots=False)
#eig_results.plot()

#%%
# Determine frequency response matrices
rslts = my_modal_sys.CalcFreqResponse(verbose=True)
f = rslts['f']
Gf = rslts['G_f']

#%%
fig_list = []
for j in range(7):
    
    for i in [0,1,3,5]:
        
        if i==0:
            plt_dict = PlotFrequencyResponse(f,Gf[:,i,j])
            fig = plt_dict["fig"]
            fig_list.append(fig)
            fig.suptitle("Gf[%d,%d]" % (i,j))
            ax_magnitude = plt_dict["ax_magnitude"]
            ax_phase = plt_dict["ax_phase"]
    
        else:
            PlotFrequencyResponse(f,Gf[:,i,j],
                                  ax_magnitude=ax_magnitude,
                                  ax_phase=ax_phase)

#%%
# Carry out pedestrian dynamics analysis
analysis_obj = UKNA_BSEN1991_2_walkers_joggers(modalsys_obj=my_modal_sys,
                                               mode_index=0)

results_obj = analysis_obj.run()

#%%
results_obj.PlotResults()
#results_obj.PlotResponsePSDs()

#%%
#fig = results_obj.PlotDeformed(200,dynsys_obj=TMD1,seperation=0.0)
#results_obj.AnimateResults(dynsys_obj=TMD1)