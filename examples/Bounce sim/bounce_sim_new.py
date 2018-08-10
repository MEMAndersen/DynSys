# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 19:32:48 2018

@author: whoever
"""

import numpy


import msd_chain
import tstep


# Define a very basic SDOF system
my_sys = msd_chain.MSD_Chain(M_vals=[1.0],
                             f_vals=[1.0],
                             eta_vals=[0.02])



def up_pass(t, y):
    return y[0]

def down_pass(t, y):
    
#    if t<3.0:
#        return 1.0 # no events
#    else:
#        return y[0]
    return y[0]-0.2

def up_pass2(t, y):
    return y[0]-0.5

def rebound(t,x):
    
    N = int(len(x)/2)
    y = x[:N]
    ydot = x[N:]
    
    ydot = -0.9*ydot
    
    return numpy.hstack((y,ydot))
    
# Attributes for hit_ground function
down_pass.terminal = True
down_pass.direction = -1

up_pass.terminal = False
up_pass.direction = +1

up_pass2.terminal = False
up_pass2.direction = +1

my_event_funcs = [up_pass,down_pass,up_pass2]
my_post_event_funcs = [None,rebound,None]

# Define time-stepping analysis
analysis = tstep.TStep(dynsys_obj=my_sys,tEnd=5.0,x0=[1.0,0.0],
                       event_funcs=my_event_funcs,
                       dt=0.01,
                       post_event_funcs=my_post_event_funcs,
                       max_events=None)

rslts = analysis.run()
fig_list = rslts.PlotResults()

fig = fig_list[0][0]
ax = fig.get_axes()[1]

t_events = rslts.t_events

for t, event_func in zip(rslts.t_events,my_event_funcs):
    
    ax.plot(t,numpy.zeros_like(t),'.',label=event_func.__name__)
    
ax.legend()

#
#

#
