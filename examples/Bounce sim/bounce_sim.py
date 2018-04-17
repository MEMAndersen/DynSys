# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:18:56 2018
An example to demonstrate use of the Scipy v1.00 ODE integration routine
Simple bounce event, with chatter sequence detection
@author: rihy
"""

import scipy.integrate
import matplotlib.pyplot as plt
import numpy as npy
import timeit

def upward_cannon(t, y):
    return [y[1], -0.5]

def hit_ground(t, y):
    return y[0]

# Attributes for hit_ground function
hit_ground.terminal = True     # defines whether integration should be terminated
hit_ground.direction = -1

# Run sim
tmin = 0.0
tmax = 100.0
dt = 2.0
y0 = npy.array([0,10.0],dtype=float)
method = 'RK45'

r = 0.56   # coefficient of restitution

terminateSolver = False
solvecount = 0
bouncecount = 0
tic=timeit.default_timer()

while not terminateSolver:
    
    # Run solution
    sol = scipy.integrate.solve_ivp(upward_cannon, [tmin, tmax], y0, method=method, events=hit_ground, max_step = dt)
    solvecount = solvecount + 1

    # Append results to arrays
    if solvecount == 1:
        t = sol.t
        y = sol.y
    else:       
        t = npy.append(t,sol.t)
        y = npy.append(y,sol.y,axis=1)
        
    # Set initial conditions (for bounce event)
    if sol.status == 1:  # termination event occurred, i.e. bounce
        
        # Register new bounce event
        bouncecount = bouncecount + 1
        if bouncecount == 1:
            tbounce = sol.t_events[0]
        else:
            tbounce = npy.append(tbounce,sol.t_events[0])
        
        # Set new initial conditions
        tmin = tbounce[-1]
        y0[0] = y[0,-1]
        y0[1] = -r*y[1,-1]
        
        # Check interval between bounce events
        if bouncecount > 1:
            if tbounce[-1] - tbounce[-2] < dt:
                terminateSolver = True
                print("Time out at t = %.2f: chatter bound sequence detected" % tbounce[-1])
    
    elif sol.status == 0: # The solver successfully reached the interval end
        terminateSolver = True
        print(sol.message) 
        
    else:
        raise ValueError("Integration step failed.")
        
toc=timeit.default_timer()
print("Solution time: %.3f seconds" % (toc-tic))
    
# Plot results
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle("Bounce simulation")

ax1.plot(t,y[0,:],'k')
ax1.set_ylabel("Displacement")
ax1.set_ylim([0,ax1.get_ylim()[1]])
ax1.set_xlim([0,tmax])

ax2.plot(t,y[1,:],'k')

for _tbounce in tbounce:
    ax1.axvline(_tbounce,color="r",linewidth=0.5)
    ax2.axvline(_tbounce,color="r",linewidth=0.5)

ax2.set_xlabel("Time")
ax2.set_ylabel("Velocity")

fig.savefig("bounce_sim.png")
