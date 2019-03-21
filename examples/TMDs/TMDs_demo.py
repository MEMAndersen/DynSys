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
modal_params['Freq'] = 0.88
modal_params['DampingRatio'] = 0.007
modal_sys = modalsys.ModalSys(name='Main system',
                              modalParams_dict=modal_params)
modal_sys.add_outputs()

# Get frequency response of systems without TMDs
freq_response_rslts = modal_sys.calc_freq_response(fmax=2.0)
   
# Get maximum ordinate of frequency response relating modal force to modal disp
max_Gf_noTMD = npy.abs(npy.max(freq_response_rslts.Gf[0,0]))

#%%

def RRF_given_TMDs(M=[5000,4800],
                   f=[0.88,0.92],
                   eta=[0.15,0.05],
                   Xpos=[50.0,60.0],
                   make_plots=False):
    """
    Function to determine response reduction factor, given TMDs defined via 
    lists
    """
    
    # Take a copy of modal system
    modal_sys_with_TMDs = deepcopy(modal_sys)
    
    # Define TMDs
    TMD_list = [TMD(name='TMD#%d' % (i+1),
                    sprung_mass=_M,
                    nat_freq=_f,
                    damping_ratio=_d,
                    verbose=False)
                for i, (_M,_f,_d) in enumerate(zip(M,f,eta))]
    
    
    # Append TMDs to modal system
    for TMD_obj, _X in zip(TMD_list,Xpos):
        modal_sys_with_TMDs.append_system(TMD_obj,Xpos_parent=_X,DOF_child=0)
    
    if make_plots:
        modal_sys_with_TMDs.plot_modeshapes()
    
    #    #%%
    #    
    #    # Get eigenvalues of system with TMDs
    #    eig_rslts = modal_sys.calc_eigenproperties()
    #    eig_rslts.plot()
    #    
    
    # Get frequency response of systems with and without TMDs
    freq_response_rslts = modal_sys_with_TMDs.calc_freq_response(fmax=2.0)
    if make_plots:
        freq_response_rslts.plot()
    
    # Get maximum ordinate of frequency response relating modal force to modal disp
    max_Gf = npy.abs(npy.max(freq_response_rslts.Gf[0,0]))
    RRF = max_Gf_noTMD / max_Gf
    
    return RRF

print(RRF_given_TMDs(make_plots=True))

#%%

# Define grid of damping values to consider
eta_vals = npy.arange(0.00,0.16,0.005)

RRF = [RRF_given_TMDs(eta=[x],M=[5000],f=[0.87],Xpos=[48.0]) for x in eta_vals]

#%%
fig,ax = plt.subplots()
ax.plot(eta_vals,RRF)
ax.set_xlabel(r"TMD damping ratio, $\xi_{TMD}$")
ax.set_ylabel("RRF")
ax.set_title("Variation of response reduction\n" +
             "factor with TMD damping ratio")
ax.set_xlim(eta_vals[0],eta_vals[-1])
