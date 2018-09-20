# -*- coding: utf-8 -*-
"""
Runs train analysis
"""

import dyn_analysis
import loading
import modalsys
import os
import numpy
import matplotlib.pyplot as plt

    
#%%
# Define system to analyse
bridge_sys = modalsys.ModalSys()
bridge_sys.AddOutputMtrx()

"""
Determine reasonable time step to use for results presentation
Note: only influences time interval at which results are output
Scipy's solver determines its own time step for solving ODEs
"""
max_fn = numpy.max(bridge_sys.fn)
dt_reqd = 0.01#(1/max_fn)/2 # sampling freq to be 10x maximum modal freq - rule of thumb for accurately capturing peaks

#%%

## Run moving load analysis for specified speed
speed_kmph=450
loading_obj = loading.LoadTrain(fName="train_defs/trainA6.csv",name="trainA6")
        


ML_analysis = dyn_analysis.MovingLoadAnalysis(modalsys_obj=bridge_sys,
                                              dt=dt_reqd,
                                              max_dt=dt_reqd,
                                              loadVel=speed_kmph*1000/3600,
                                              loadtrain_obj=loading_obj,
                                              tEpilogue=1.0,
                                              writeResults2File=True)
                                              #results_fName="my_results.csv")

results_obj = ML_analysis.run()

#%%
# Plot both state and response results
results_fig = results_obj.PlotResults()

#%%
# Plot just state results (i.e. modal results)
results_fig = results_obj.PlotStateResults()

#%%
# Plot just response results
results_fig = results_obj.PlotResponseResults()

#%%
# Produce animation of displacement results
anim = results_obj.AnimateResults(SysPlot_kwargs={'y_lim':(-0.005,0.005)},
                                  FuncAnimation_kwargs={'repeat':True,
                                                        'repeat_delay':1000})

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
trains2analyse = trains2analyse[:1] # use to select certain trains

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

#%%
def PlotStats(trainSpeeds_kmph,
              trainCodes,
              statsData,
              responseNames,
              ylim=[0,5.0],
              xlabel="Train speed [km/hr]",
              supTitle=""):
    """
    statsData is expected to be ndarray
        axis 0: speeds
        axis 1: trains
        axis 2: responses
    """
    
    # Convert types
    trainCodes=numpy.asarray(trainCodes)
    trainSpeeds_kmph=numpy.asarray(trainSpeeds_kmph)
    
    # Counters
    nSpeeds = statsData.shape[0]
    nTrains = statsData.shape[1]
    nResponses = statsData.shape[2]
    
    # Check shape of ndarray agrees with inputs
    if trainCodes.shape[0]!=nTrains:
        raise ValueError("trainCodes.shape[0]!=nTrains\n"+
                         "trainCodes.shape: {0}\n".format(trainCodes.shape)+
                         "statsData.shape: {0}".format(statsData.shape))
        
    if trainSpeeds_kmph.shape[0]!=nSpeeds:
        raise ValueError("trainSpeeds_kmph.shape[0]!=nSpeeds\n"+
                         "trainSpeeds_kmph.shape: {0}\n".format(trainSpeeds_kmph.shape)+
                         "statsData.shape: {0}".format(statsData.shape))
        
    # Create new figure
    fig, axarr = plt.subplots(nResponses, sharex=True)
    fig.set_size_inches(16,10)
    
    fig.suptitle(supTitle)
    
    for r in range(nResponses):
        
        ax = axarr[r]
    
        ax.plot(trainSpeeds_kmph,statsData[:,:,r],label=trainCodes)
        
        handles, labels = ax.get_legend_handles_labels()
        
        if r==0:
            fig.legend(handles, trainCodes, loc='lower center',fontsize='x-small',ncol=5)

        ax.set_xlim([trainSpeeds_kmph.min(),trainSpeeds_kmph.max()])
        ax.set_ylim(ylim)
        
        ax.set_title(responseNames[r],fontsize='x-small')
        
        if r==nResponses-1:
            ax.set_xlabel(xlabel)
            
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)        # create space for suptitle
    fig.subplots_adjust(bottom=0.15)      # create space for figlegend
    
    return fig, axarr

#%%
#
## Reload pickled results
#structureCode = "LRN"
#massCode = "min"
#pkl_fName = 'Multiple_MovingLoadAnalysis_{0}_{1}.pkl'.format(structureCode,massCode)
#multipleAnalyses = dyn_analysis.load(fName=pkl_fName)

#%%

# Extract data to plot
absmax_stats = multipleAnalyses.stats_dict[bridge_sys]["absmax"]
trainNames = [x.name for x in multipleAnalyses.vals2permute["loadtrain_obj"]]
speeds_kmph = (3600/1000 * numpy.array(multipleAnalyses.vals2permute["loadVel"])).tolist()
responseNames = multipleAnalyses.dynsys_obj.output_names
sys_name = multipleAnalyses.dynsys_obj.name

# Plot stats, using function defined above
fig, axarr = PlotStats(trainSpeeds_kmph=speeds_kmph,
                       trainCodes=trainNames,
                       statsData=absmax_stats,
                       responseNames=responseNames,
                       supTitle="Maximum acceleration responses for {0}".format(sys_name))
#fig.savefig("MaxAccSummary.png")


