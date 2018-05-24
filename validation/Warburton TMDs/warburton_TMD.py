# -*- coding: utf-8 -*-
"""
Validation script to compare frequency response obtained using `dynsys` 
routines against Warburton and Ayorinde's 1980 derivation concerning a 
single damped mode with a single damped TMD: 
[PDF](.../references/Warburton and Ayorinde (1980).pdf)
"""

__author__ = "Richard Hollamby, COWI UK Bridge, RIHY"

import numpy

# Modules to be tested
import msd_chain
from dynsys import PlotFrequencyResponse

#%%

fmax=3.0 # freq limit for plots

#%%

# Define two-mass system using Warburtons notation
m_M = 7344  #kg, mass of main system
m_D = 640.0  #kg, mass of damper system
f_M = 1.878 #Hz, nat freq of main system
f_D = 1.640  #Hz, nat freq of damper
gamma_M = 0.005 # damping ratio, main system
gamma_D = 0.180 # damping ratio, TMD system


d, main_sys_with_TMD, f, G_f = msd_chain.warburton_TMD(m_M=m_M,
                                                       f_M=f_M,
                                                       mu=m_D/m_M,
                                                       f=f_D/f_M,
                                                       gamma_M=gamma_M,
                                                       gamma_A=gamma_D,
                                                       fmax=fmax)

# Define output matrix
main_sys_with_TMD.AddOutputMtrx(fName='outputs_withTMDs.csv')

#%%

plot_dict = PlotFrequencyResponse(f,G_f,
                                  label_str="per Warburton & Ayorinde eqn(1)")

print("Max |G_f|, system with TMDs: %.2e" % numpy.max(numpy.abs(G_f)))

#%%

# Obtain frequency response using CalcFreqResponse()
f2 , G_f2 = main_sys_with_TMD.CalcFreqResponse(fmax=fmax)


# Overlay to compare frequency response
plot_dict = PlotFrequencyResponse(f2,G_f2[0,0,:],
                                  label_str="using CalcFreqResponse()",
                                  ax_magnitude=plot_dict["ax_magnitude"],
                                  ax_phase=plot_dict["ax_phase"])

print("Max |G_f|, system with TMDs: %.2e" % numpy.max(numpy.abs(G_f2[0,0,:])))

# Edit line styles
lines2edit = [plot_dict["ax_magnitude"].lines[1],
              plot_dict["ax_phase"].lines[1]]

for line in lines2edit:
    line.set_linestyle("--")


#%%
    
# Overlay frequency transfer function for relative displacement
plot_dict = PlotFrequencyResponse(f2,G_f2[2,0,:],
                              label_str="Relative disp (m)",
                              ax_magnitude=plot_dict["ax_magnitude"],
                              ax_phase=plot_dict["ax_phase"])

print("Max |G_f|, relative disp: %.2e" % numpy.max(numpy.abs(G_f2[2,0,:])))

plot_dict["ax_magnitude"].axvline(f_D,color='c',alpha=0.4)
plot_dict["ax_phase"].axvline(f_D,color='c',alpha=0.4)

#%%

# Obtain frequency response for main mass only
main_sys = msd_chain.MSD_Chain(M_vals = m_M,
                               f_vals = f_M,
                               eta_vals = gamma_M)

f3 , G_f3 = main_sys.CalcFreqResponse(fmax=fmax)

print("Max |G_f|, system with no TMDs: %.2e" % numpy.max(numpy.abs(G_f3[0,0,:])))

# Overlay to compare frequency response
PlotFrequencyResponse(f3,G_f3[0,0,:],
                      label_str="using CalcFreqResponse(), no TMD",
                      ax_magnitude=plot_dict["ax_magnitude"],
                      ax_phase=plot_dict["ax_phase"])



                      

