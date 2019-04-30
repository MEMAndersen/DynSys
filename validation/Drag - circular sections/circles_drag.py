# -*- coding: utf-8 -*-
"""
Validate script, to demonstrate accuracy of `calc_C_d()` method for circles, 
by re-creating Fig 7.28, BS EN 1991-1-4

RIHY, 19/02/2019
"""
    
import numpy
import matplotlib.pyplot as plt
from wind_section import WindSection_Circular, calc_U_from_Re

#%%
b=0.1 #arbitrary

# List k/b, Re values per figure
k_b_vals = numpy.array([1e-2,1e-3,1e-4,1e-5,1e-6])
Re_vals = numpy.logspace(5,7,num=100)

# Convert to inputs required
k_vals =  k_b_vals * b
U_vals = calc_U_from_Re(d=b, Re=Re_vals)

# Evaluate c_f0 for each k,U pair
outer_list = []

for k in k_vals:
    
    inner_list = []
    
    for U in U_vals:
        
        obj = WindSection_Circular(d=b,k=k)
        c_f0 = obj.calc_C_D(U)
        inner_list.append(c_f0)
        
    outer_list.append(inner_list)
    
c_f0 = numpy.array(outer_list)

# Re-create Fig. 7.28 to test calc_C_d() method
fig, ax = plt.subplots()

h = ax.plot(Re_vals,c_f0.T)
h[-1].set_linestyle('--')

ax.set_xscale('log')

ax.legend(h,["%.0e" % x for x in k_b_vals],title="$k/b$")

ax.set_ylim([0,1.4]) # per Fig 7.28
ax.set_xlim([Re_vals[0],Re_vals[-1]]) # per Fig 7.28

ax.set_xlabel("$Re$")
ax.set_ylabel("$c_{f0}$")
ax.set_title("Drag coefficients for circles\n" + 
             "according to Fig 7.28, BS EN 1991-1-4:2005")