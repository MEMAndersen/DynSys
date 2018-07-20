
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
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import scipy

# DynSys imports
import dynsys
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

#%%

# Evaluate modal damping matrix based on 1 person on bridge
C_pa = ped_dyn.calc_modal_damping_latsync(bridge_sys)

#%%
    
# Run analysis for various pedestrian numbers
# to determine how eigenproperties change
s_vals = []
eta_vals = []
fd_vals = []
Np_vals = numpy.arange(0,2000,10)
MAC_vals = []

C_bridge = bridge_sys._C_mtrx

for i, Np in enumerate(Np_vals):
    
    # Adjust bridge damping matrix
    bridge_sys._C_mtrx = C_bridge - Np * C_pa

    # Carry out eigevalue analysis
    eig_props = bridge_sys.CalcEigenproperties()
    
    # Record key results of interest
    s_vals.append(eig_props["s"])
    eta_vals.append(eig_props["eta"])
    fd_vals.append(eig_props["f_d"])
    
# Restore with original bridge-only damping matrix
bridge_sys._C_mtrx = C_bridge
    
#%%

# Define figure to provide summary of results
fig, axarr = plt.subplots(2,2)
fig.set_size_inches((14,7))
fig.subplots_adjust(hspace=0.5)
    
# Plot damping ratio against damped natural frequency (per Figure 5 in paper)
ax = axarr[0,0]

ax.plot(eta_vals,fd_vals,'k.',markersize=0.5)
ax.plot(eta_vals[::10],fd_vals[::10],'ko',markersize=1.5)
ax.plot(eta_vals[0],fd_vals[0],'bo',markersize=3.0)

ax.set_xlim([-0.02,0.22]) # per McRobie paper
ax.set_ylim([0.8,1.05])   # per McRobie paper
ax.axvline(x=0.0,color='r',alpha=0.3)
ax.set_xlabel("Damping ratio")
ax.set_ylabel("Damped natural frequency (Hz)")
ax.set_title("Frequency vs Effective Damping\n"+
             "(per Figure 5 of McRobie's paper)")

#%%

# Plot poles (per Figure 4 in paper)
ax = axarr[0,1]

ax.plot(numpy.real(s_vals),numpy.imag(s_vals),'k.',markersize=0.5)
ax.plot(numpy.real(s_vals[::10]),numpy.imag(s_vals[::10]),'ko',markersize=1.5)
ax.plot(numpy.real(s_vals[0]),numpy.imag(s_vals[0]),'bo',markersize=3.0)

ax.axvline(x=0.0,color='r',alpha=0.3)

ax.set_xlim([-1.2,+0.2]) # per McRobie paper
ax.set_ylim([-8.0,+8.0])   # per McRobie paper

ax.set_xlabel("Real(s)")
ax.set_ylabel("Imag(s)")
ax.set_title("Eigenvalues of A; variation with Np\n"+
             "(per Figure 4 of McRobie's paper)")

#%%
# Plot damping ratio vs number of pedestrians
ax = axarr[1,0]

ax.plot(Np_vals,eta_vals,'k.',markersize=0.5)

ax.axhline(y=0.0,color='r',alpha=0.3)

ax.set_xlim([0,2000])

ax.set_xlabel("Number of pedestrians")
ax.set_ylabel("Effective damping ratio")
ax.set_title("Effect of pedestrians on damping")

#%%
ax = axarr[1,1]
ax.set_visible(False)

#%%
fig.savefig('results_plots.png')