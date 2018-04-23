# -*- coding: utf-8 -*-
"""
Script to illustrate pedestrian dynamics analyses to UK NA to BS EN 1991-2
"""

# DynSys package imports
import modalsys
from dyn_analysis import UKNA_BSEN1991_2_walkers_joggers as walkers_joggers_analysis
from dyn_analysis import PedestrianDynamics_transientAnalyses

#%%

bridgeClass='b'

# Define modal system
my_sys = modalsys.ModalSys(name="Bridge example")
#my_sys.PrintSystemMatrices()

# Add output matrix
my_sys.AddOutputMtrx()

#%%

# Define single walkers/joggers analysis
my_analysis = walkers_joggers_analysis(modalsys_obj=my_sys,
                                       mode_index=4,
                                       analysis_type="joggers",
                                       bridgeClass=bridgeClass)
                                       
results_obj = my_analysis.run()
tstep_obj = results_obj.tstep_obj

results_obj.PlotStateResults()
results_obj.PlotResponseResults(responses2plot=[0,1])
results_obj.PlotResponseResults(responses2plot=[2,3])

#%%

# Run all walkers/joggers analyses
all_analyses = PedestrianDynamics_transientAnalyses(modalsys_obj=my_sys,
                                                    bridgeClass=bridgeClass)
all_analyses.run(save=False)

#%%
all_analyses.plot_response_stats()

all_analyses.plot_response_stats(dynsys2plot=my_sys,
                                 responses2plot=[2,3],
                                 sharey=True)

#%%
all_analyses.plot_modal_params()