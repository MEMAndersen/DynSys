
# coding: utf-8

"""
Validation script to demonstrate multiple modes + multiple TMDs analysis method

Comparison is made against results and figures presented in the following 
paper: 
[PDF](.../references/The Lateral Dynamic Stablity of Stockton Infinity Footbridge using Complex Modes.pdf)    
"""

__author__ = "Richard Hollamby, COWI UK Bridge, RIHY"

# Standard imports
import numpy
import matplotlib.pyplot as plt

# DynSys imports
import modalsys
import damper
import ped_dyn

plt.close('all')

#%%

# Define modal system
bridge_sys = modalsys.ModalSys(name="Stockton Infinity Footbridge",
                               fname_modeshapes='deck_modeshapes.csv',
                               fname_modalParams='modal_params.csv')

#%%

# Define TMD systems and append to bridge system
damper.append_TMDs(modal_sys=bridge_sys,
                   fname="TMD_defs.csv",
                   append=True,
                   verbose=True)

fig = bridge_sys.PlotModeshapes()[0]
fig.savefig('modeshapes.png')

#%%

# Define and run lat sync analysis
latsync_analysis = ped_dyn.LatSync_McRobie(bridge_sys)
latsync_analysis.run(Np_vals=numpy.arange(0,2500,10))

#%%

# Plot results
fig = latsync_analysis.plot_results()
fig.savefig('results_plots.png')

#%%

# Manually scale axes to allow comparison against figures in McRobie paper

ax_list = fig.get_axes()

ax=ax_list[0]
ax.set_xlim([-0.02,0.22])   # per McRobie paper
ax.set_ylim([0.8,1.05])     # per McRobie paper

ax=ax_list[1]
ax.set_xlim([-1.2,+0.2])    # per McRobie paper
ax.set_ylim([-8.0,+8.0])    # per McRobie paper

