# -*- coding: utf-8 -*-
"""
Example of response spectrum analysis

The classic El Centro earthquake is used by way of example

Time series data obtained from here:
http://www.vibrationdata.com/elcentro.html

Results for comparison obtained from here:
https://www.mathworks.com/examples/matlab/community/
20022-chopra-2012-elastic-response-spectrum-for-elcentro-earthquake
-california-may-18-1940-ns-component

"""

import scipy
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from dyn_analysis import ResponseSpectrum

plt.close("all")

#%%

# Read in ground motion time series
ground_acc = pd.read_csv('el-centro.csv',header=0,index_col=0)
t = ground_acc.index
tmax = t[-1]
a = numpy.ravel(ground_acc.values) * 9.81  # convert to m/s2
dt = t[1]-t[0]
max_a = numpy.max(numpy.abs(a))

fig, ax = plt.subplots()
ax.plot(t,a)
ax.set_xlim([0,31.0])
ax.set_xlabel("Time (secs)")
ax.set_ylabel("Acceleration (m/$s^2$)")
ax.set_title("Ground acceleration recorded during\n"+
             "El-Centro earthquake (May 18th 1940)\n"+
             "North-South component")
fig.tight_layout()

# Use Scipy to obtain interpolation function based on input acceleration time series
accFunc = scipy.interpolate.interp1d(t,a,bounds_error=False,fill_value=0.0)

#%%

# Compute response spectrum
T_vals = numpy.logspace(-2,1.7,400)

eta = 0.1

results = ResponseSpectrum(accFunc,
                           tResponse=tmax,
                           max_dt=dt,
                           T_vals=T_vals,
                           eta=eta)

#%%
# Make plot look similar to reference plot
fig = results["fig"]
fig.set_size_inches((8,8))
ax3 = fig.get_axes()[2]
ax3.set_xlim([0.01,50.0])
ax3.set_xscale('log')
ax3.set_xlabel("$T_N$, seconds")
ax3.axhline(max_a,color='r',alpha=0.3)
fig.savefig("calculated.png")


