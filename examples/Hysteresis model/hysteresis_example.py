# Hysteresis model
# http://eprints.lancs.ac.uk/1375/1/MFI_10c.pdf
# Identification of Hysteresis Functions Using a Multiple Model Approach
# Mihaylova, Lampaert et al

import numpy as npy
import matplotlib.pyplot as plt

from hysteresis import HysteresisModel, static_response

plt.close('all')

#%%
  
# Define hysteresis model
K = [0.1,0.8,10.0,10.0]
W = [1.0,1.0,1.0,0.5]
Ne = len(K)
hys = HysteresisModel(Ne,K,W=W)

# Read force function from file
fname = 'LW1_1607_MONTH.csv'
data = npy.genfromtxt(fname,delimiter=',',skip_header=1)
t_vals = data[:,0]
disp_vals = -data[:,1]
disp_vals = disp_vals - disp_vals[0]
F_vals = data[:,2]
F_vals = F_vals - F_vals[0]
plt.plot(disp_vals,F_vals)

#%%
# Define spring
K_spring = 1

# Define and run analysis
analysis = static_response(hys_obj=hys,K_spring=K_spring)
analysis.run(F_vals=F_vals)
fig = analysis.plot()   

axarr = fig.get_axes()
ax = axarr[2]
ax.plot(disp_vals,label='measured')
ax.legend()