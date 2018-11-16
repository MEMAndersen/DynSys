# Hysteresis model
# http://eprints.lancs.ac.uk/1375/1/MFI_10c.pdf
# Identification of Hysteresis Functions Using a Multiple Model Approach
# Mihaylova, Lampaert et al

import numpy as npy
import matplotlib.pyplot as plt
#from scipy.optimize import minimize

from hysteresis import HysteresisModel, static_response

plt.close('all')

#%%

# Read force function from file
fname = 'LW1_1607_MONTH.csv'
data = npy.genfromtxt(fname,delimiter=',',skip_header=1)
disp_vals = -data[:,1]
disp_vals = disp_vals - disp_vals[0]
F_vals = data[:,2]
F_vals = F_vals - F_vals[0]
plt.plot(disp_vals,F_vals)

#%%

def run_sim(x,verbose=True,make_plot=False):
      
    # Define hysteresis model
    x = list(x)
    K1 = x.pop(0)
    K2 = x.pop(0)
    Ne = int(len(x)/2)
    K = x[:Ne]
    W = x[Ne:]
    hys = HysteresisModel(Ne,K,W=W)
            
    # Define and run analysis
    analysis = static_response(hys_obj=hys,K1=K1,K2=K2)
    analysis.run(F_vals=F_vals)
    
    if make_plot:
        fig = analysis.plot()
    
    # Calculate mean-squared error
    predicted_disp_vals = analysis.u_vals
    disp_error = disp_vals - predicted_disp_vals
    RMS_error = (sum(disp_error**2) / len(disp_error))**0.5
    
    # Overall meaasured displacements
    if make_plot:
        ax_list = fig.get_axes()
        ax = ax_list[2]
        ax.plot(disp_vals,label='measured')
        ax.legend()
        
        ax = ax_list[0]
        ax.plot(analysis.F_net_vals,label='error')
        
        ax = ax_list[5]
        ax.plot(disp_vals,F_vals,label='measured')
        ax.legend()
        
    if verbose:
        print("K1:\t{0}".format(K1))
        print("K2:\t{0}".format(K2))
        print("K:\t\t{0}".format(K))
        print("W:\t\t{0}".format(W))
        print("RMS_error:\t%.3f" % RMS_error)
        print("---")
        
    return RMS_error

#%%
    
K1 = 1.0
K2 = 2.0
K_vals = [0.1,1.0,10.0]
W_vals = [1.0,1.0,1.0]
run_sim([K1,K2,*K_vals,*W_vals],make_plot=True)

#%%
#
#x = minimize(run_sim,x0=[K_spring,*K_vals,*W_vals],
#             method='Powell',
#             options={'disp':True})