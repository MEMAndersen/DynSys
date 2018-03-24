# -*- coding: utf-8 -*-
"""
Example function associated with `modalsys.py`
***

Function defines two identical modal systems (using input from file). To the 
first a system of 3no TMDs is appended. The effect this has is illustrated by
obtaining the frequency response G(f) for both systems

"""

# Imports
import numpy as npy

import modalsys
from dynsys import PlotFrequencyResponse


# -------------- EXAMPLE SCRIPT --------------------

# Define ModalSys instances, reading definition data from files
modal_sys = modalsys.ModalSys()
modal_sys2 = modalsys.ModalSys()

# Define TMD properties
X_TMD = [41.2,48.8,56.3]                              # position along deck
mass_TMD = [5000,4800,4900]                           # TMD masses in kg
freq_TMD = npy.asmatrix([0.88,0.88,0.88])             # tuned frequencies for TMDs
eta_TMD = [0.2,0.2,0.2]                               # dampign ratios for TMDs in isolation

# Append TMDs to modal system
modal_sys.AppendTMDs(X_TMD,mass_TMD,freq_TMD,eta_TMD)

# Plot TMD locations on modeshapes
ax = modal_sys.PlotModeshapes(num=100,L=180,plotTMDs=True)

# Define output matrix to obtain modal displacements
N = modal_sys.nDOF
output_modeDisps = npy.asmatrix(npy.hstack((npy.identity(N),npy.zeros((N,2*N)))))
modal_sys.AddOutputMtrx(output_mtrx=output_modeDisps)

N = modal_sys2.nDOF
output_modeDisps = npy.asmatrix(npy.hstack((npy.identity(N),npy.zeros((N,2*N)))))
modal_sys2.AddOutputMtrx(output_mtrx=output_modeDisps)

# Get frequency response of systems with and without TMDs
f, G_f = modal_sys.CalcFreqResponse(fmax=2.0)
f_2, G_f_2 = modal_sys2.CalcFreqResponse(fmax=2.0)

# Get damped natural frequencies (poles) of systems with and without TMDs
f_d = modal_sys.CalcEigenproperties()["f_d"]
f_d_2 = modal_sys2.CalcEigenproperties()["f_d"]

# Plot frequency responses
dof_i = 0
dof_j = 0

plots = PlotFrequencyResponse(f,G_f[dof_i,dof_j,:],label_str="With TMDs")

ax1 = plots["ax_magnitude"]
ax2 = plots["ax_phase"]

PlotFrequencyResponse(f_2,G_f_2[dof_i,dof_j,:],
                      label_str="No TMDs",
                      ax_magnitude=ax1, ax_phase=ax2,
                      f_d=f_d_2)