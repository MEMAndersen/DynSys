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
import damper
from dynsys import PlotFrequencyResponse

#%%

fmax=3.0 # freq limit for plots

#%%

# Define two-mass system using Warburtons notation
m_M = 7344  #kg, mass of main system
m_D = 640.0  #kg, mass of damper system
f_M = 1.878 #Hz, nat freq of main system
f_D = 1.878  #Hz, nat freq of damper
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
rslts = main_sys_with_TMD.CalcFreqResponse(fmax=fmax)
f2 = rslts['f']
G_f2 = rslts['G_f']


# Overlay to compare frequency response
plot_dict = PlotFrequencyResponse(f2,G_f2[:,0,0],
                                  label_str="using CalcFreqResponse()",
                                  ax_magnitude=plot_dict["ax_magnitude"],
                                  ax_phase=plot_dict["ax_phase"])

print("Max |G_f|, system with TMDs: %.2e" % numpy.max(numpy.abs(G_f2[0,0,:])))

# Edit line styles
lines2edit = [plot_dict["ax_magnitude"].lines[-1],
              plot_dict["ax_phase"].lines[-1]]

for line in lines2edit:
    line.set_linestyle("--")


#%%
    
# Overlay frequency transfer function for relative displacement
plot_dict = PlotFrequencyResponse(f2,G_f2[:,2,0],
                              label_str="Relative disp (m)",
                              ax_magnitude=plot_dict["ax_magnitude"],
                              ax_phase=plot_dict["ax_phase"])

print("Max |G_f|, relative disp: %.2e" % numpy.max(numpy.abs(G_f2[:,2,0])))

plot_dict["ax_magnitude"].axvline(f_D,color='c',alpha=0.4)
plot_dict["ax_phase"].axvline(f_D,color='c',alpha=0.4)

#%%

# Obtain frequency response for main mass only
main_sys = msd_chain.MSD_Chain(M_vals = m_M,
                               f_vals = f_M,
                               eta_vals = gamma_M)

rslts = main_sys.CalcFreqResponse(fmax=fmax)
f3 = rslts['f']
G_f3 = rslts['G_f']

print("Max |G_f|, system with no TMDs: %.2e" % numpy.max(numpy.abs(G_f3[:,0,0])))

# Overlay to compare frequency response
PlotFrequencyResponse(f3,G_f3[:,0,0],
                      label_str="using CalcFreqResponse(), no TMD",
                      ax_magnitude=plot_dict["ax_magnitude"],
                      ax_phase=plot_dict["ax_phase"])

#%%

# Define damper mass as seperate system
TMD_sys = damper.TMD(sprung_mass=m_D,nat_freq=f_D,damping_ratio=gamma_D)

# Append to main system
main_sys.AppendSystem(child_sys=TMD_sys,DOF_parent=0,DOF_child=0)

# Calculate frequency response of combined system
rslts = main_sys.CalcFreqResponse(fmax=fmax)
f4 = rslts['f']
G_f4 = rslts['G_f']

# Overlay to compare frequency response
PlotFrequencyResponse(f4,G_f4[:,0,0],
                      label_str="Relative disp, using CalcFreqResponse(); " + 
                                "TMD as appended system",
                      ax_magnitude=plot_dict["ax_magnitude"],
                      ax_phase=plot_dict["ax_phase"])

# Edit line styles
lines2edit = [plot_dict["ax_magnitude"].lines[-1],
              plot_dict["ax_phase"].lines[-1]]

for line in lines2edit:
    line.set_linestyle("--")
    
#%%
    
# Define main system as a 2dof system but constraint dof0 to ground
main_sys_constrained = damper.TMD(sprung_mass=m_M,
                                  nat_freq=f_M,
                                  damping_ratio=gamma_M)

main_sys_constrained.AddConstraintEqns(Jnew=[1,0],Jkey="Constrain to ground")

# Calculate frequency response of combined system
rslts = main_sys_constrained.CalcFreqResponse(fmax=fmax)
f5 = rslts['f']
G_f5 = rslts['G_f']

# Overlay to compare frequency response
PlotFrequencyResponse(f5,G_f5[:,0,1],
                      label_str="using CalcFreqResponse(), no TMD; " + 
                                "2dof constrained system",
                      ax_magnitude=plot_dict["ax_magnitude"],
                      ax_phase=plot_dict["ax_phase"])

# Edit line styles
lines2edit = [plot_dict["ax_magnitude"].lines[-1],
              plot_dict["ax_phase"].lines[-1]]

for line in lines2edit:
    line.set_linestyle("--")


                      

