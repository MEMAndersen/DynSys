# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:42:44 2018

@author: whoever
"""

import numpy
import matplotlib.pyplot as plt

plt.close('all')

# DynSys package imports
import modalsys
import tstep
from damper import TMD
from msd_chain import MSD_Chain
from ped_dyn import UKNA_BSEN1991_2_walkers_joggers

#%%

# Define modal system
my_sys = modalsys.ModalSys(name="Bridge example")

# Define a system to append
TMD1 = TMD(sprung_mass=2000,nat_freq=1.9,damping_ratio=0.1,name='TMD1')

# Statically append TMDs to modal system
my_sys.AppendSystem(child_sys=TMD1,Xpos_parent=58.0,DOF_child=0)

#my_sys.PlotModeshapes()

#%%
# Define single walkers/joggers analysis
my_analysis = UKNA_BSEN1991_2_walkers_joggers(modalsys_obj=my_sys,
                                              mode_index=1,
                                              analysis_type="joggers")
                                       
results_obj = my_analysis.run()

#%%
#anim = results_obj.AnimateResults()

results_obj.PlotStateResults(my_sys)

#%%

modeshape_func = my_sys.modeshapeFunc
velocity = 3.0

def constraints_func(t):
    
    x = velocity*t
    J = numpy.asmatrix(modeshape_func(x))
    
    return J

constraints_func(t=5.0)

#%%
J = my_sys._J_dict['0_1']
my_sys._J_dict['0_1'] = constraints_func

results_obj = my_analysis.run()
results_obj.PlotStateResults(my_sys)

#%%

# Explore how eigenproperties of system vary with time

t_vals = numpy.arange(0,61,3)
fn_vals = []
eta_vals = []

for _t in t_vals:
    
    eig_props = my_sys.CalcEigenproperties(t=_t)
    fn_vals.append(eig_props['f_n'][::2])
    eta_vals.append(eig_props['eta'][::2])
 
fn_vals = numpy.array(fn_vals)
eta_vals = numpy.array(eta_vals)

fig,axarr = plt.subplots(2,sharex=True)

ax = axarr[0]
ax.plot(t_vals,fn_vals-fn_vals[0])
ax.set_ylabel("Shift in natural frequency (Hz)")

ax = axarr[1]
h=ax.plot(t_vals,eta_vals)
ax.set_xlabel("Time (secs)")
ax.set_ylabel("Damping ratio")
ax.set_xlim([0,60.0])

fig.suptitle("Variation in modal properties with time")
fig.legend(h,["Mode %d" % (i+1) for i in range(len(h))])

#%%

# Define a new system
msd_sys = MSD_Chain(M_vals = [1000,2000,500],
                    f_vals = [1.0,1.0,1.00],
                    eta_vals = [0.02,0.04,0.01])
msd_sys.AddConstraintEqns(Jnew=numpy.asmatrix([[1,-1,-2]]),Jkey='0')

J = msd_sys._J_dict['0']

def constraints_func2(t,omega = 2.0):
    temp = J
    temp[0,0] = numpy.sin(2*numpy.pi*omega*t)
    return temp

msd_sys._J_dict['0']=constraints_func2

#%%
analysis_obj = tstep.TStep(msd_sys,x0=[1,0,0,0,0,0])
results_obj = analysis_obj.run()
results_obj.PlotStateResults()

#%%
anim = results_obj.AnimateResults()