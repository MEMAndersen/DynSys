# -*- coding: utf-8 -*-
"""
Runs train analysis
"""

import numpy
import matplotlib.pyplot as plt
import os

import dyn_analysis
import loading
import modalsys
    
#%%
# Define system to analyse
bridge_sys = modalsys.ModalSys(name='My Bridge')
bridge_sys.AddOutputMtrx()

"""
Determine reasonable time step to use for results presentation
Note: only influences time interval at which results are output
Scipy's solver determines its own time step for solving ODEs
"""
max_fn = numpy.max(bridge_sys.modalParams_dict['Freq'])
dt_reqd = 0.01#(1/max_fn)/2 # sampling freq to be 10x maximum modal freq - rule of thumb for accurately capturing peaks

#%%

## Run moving load analysis for specified speed
train_code='trainA6'
speed_kmph=450
loading_obj = loading.LoadTrain(fName="train_defs/%s.csv" % train_code,
                                name=train_code)
        
analysis_name = '%s, %.0f kmph' % (train_code,speed_kmph)

ML_analysis = dyn_analysis.MovingLoadAnalysis(modalsys_obj=bridge_sys,
                                              name=analysis_name,
                                              dt=dt_reqd,
                                              max_dt=dt_reqd,
                                              loadVel=speed_kmph*1000/3600,
                                              loadtrain_obj=loading_obj,
                                              tEpilogue=1.0,
                                              writeResults2File=False,
                                              results_fName="my_results.csv")

results_obj = ML_analysis.run()

#%%
# Plot both state and response results
results_fig = results_obj.plot_results()

#%%
# Plot just state results (i.e. modal results)
results_fig = results_obj.plot_state_results()

#%%
# Plot just response results
results_fig = results_obj.plot_response_results()

#%%
# Produce animation of displacement results
anim = results_obj.AnimateResults(SysPlot_kwargs={'y_lim':(-0.005,0.005)},
                                  fig_size=(14,6),
                                  FuncAnimation_kwargs={'repeat':True,
                                                        'repeat_delay':1000})

#%%
#from matplotlib.animation import PillowWriter
#anim.save("%s.gif" % train_code, writer=PillowWriter(fps=15))

#%%
# Plot periodogram (power spectral density estimate) of results
results_obj.PlotResponsePSDs()


#%%
# Run multiple analyses for various trains and speeds

# Define speeds to consider
speeds_kmph = numpy.arange(300,451,50)
kmph_to_mps = 1000/3600 # conversion factor

# Define loading patterns
train_defs_path = 'train_defs/'
trains2analyse = [train_defs_path + x for x in os.listdir(train_defs_path)]
trains2analyse = trains2analyse[:2] # use to select certain trains

loading_obj_list = []
for train_fName in trains2analyse:
    loading_obj_list.append(loading.LoadTrain(fName=train_fName))
    
# Run multiple moving load analyses
multipleAnalyses = dyn_analysis.Multiple(dyn_analysis.MovingLoadAnalysis,
                                         dynsys_obj=bridge_sys,
                                         loadVel=(speeds_kmph*kmph_to_mps).tolist(),
                                         loadtrain_obj=loading_obj_list,
                                         dt=dt_reqd,
                                         tEpilogue=1.0,
                                         retainResponseTimeSeries=False,
                                         writeResults2File=False)
multipleAnalyses.run(save=False)

# Write statistics DataFrame to csv file
multipleAnalyses.stats_df.to_csv('stats_results.csv')

#%%

fig_list = multipleAnalyses.plot_stats(stat='absmax',
                                       max_responses_per_fig=4)[0]

# Customise figures
for fig in fig_list:
    
    fig.subplots_adjust(left=0.17)
    fig.set_size_inches((10,6))
    axlist = fig.get_axes()
    [ax.set_ylim([0,2.0]) for ax in axlist]
    axlist[-1].set_xlabel("Train speed (m/s)")
    fig.suptitle("Mid-span acceleration response\n" + 
                 "versus speed, for various trains")

