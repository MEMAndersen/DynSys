# -*- coding: utf-8 -*-
'''
This example demonstrates a simple use of pycallgraph.
'''

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput

import dyn_analysis
import loading
import modalsys
import os
import numpy
import matplotlib.pyplot as plt

#%%

def myFunc():
    """
    Runs train analysis
    """
    
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
    
    ## Run moving load analysis for specified speed
    speed_kmph=450
    loading_obj = loading.LoadTrain(fName="train_defs/trainA6.csv",name="trainA6")
            
    
    
    ML_analysis = dyn_analysis.MovingLoadAnalysis(modalsys_obj=bridge_sys,
                                                  dt=dt_reqd,
                                                  max_dt=dt_reqd,
                                                  loadVel=speed_kmph*1000/3600,
                                                  loadtrain_obj=loading_obj,
                                                  tEpilogue=1.0,
                                                  writeResults2File=True,
                                                  results_fName="my_results.csv")
    
    results_obj = ML_analysis.run()
    
    return results_obj

#%%


if __name__ == '__main__':
    
    graphviz = GraphvizOutput()
    #config = Config(max_depth=10)
    graphviz.output_file = 'call_graph.png'

    with PyCallGraph(output=graphviz):#, config=config):
        myFunc()