# -*- coding: utf-8 -*-
"""
Classes and methods used to store results from frequency response analysis
"""

import matplotlib.pyplot as plt
import numpy as npy


class FreqResponse_Results():
    
    def __init__(self,f,Gf,output_names=None,input_names=None):
        
        self.f = f
        """
        Array of frequency values (Hz) at which frequency response has been 
        evaluated
        """
        
        self.Gf = Gf
        """
        Ndarray of shape (Nf,No,Ni) where Nf=len(f) and [No,Ni] is frequency 
        response matrix at each frequency value considered
        """
        
        if output_names is None:
            output_names = ['Output %d' % i for i in range(Gf.shape[1])]
        self.output_names = output_names
        """
        List of output names, length No
        """
        
        if input_names is None:
            input_names = ['Input %d' % i for i in range(Gf.shape[2])]
        self.input_names = input_names
        """
        List of input names, length Ni
        """

      
    # --------------------
    @property
    def f(self):
        return self._f
    
    @f.setter
    def f(self,value):
        self._f = value
        
    # --------------------
    @property
    def Gf(self):
        return self._Gf
    
    @property
    def G_f(self):
        """
        Synonym to `Gf`
        """
        return self.Gf
    
    @Gf.setter
    def Gf(self,value):
        self._Gf = value
        
    @G_f.setter
    def G_f(self,value):
        self.Gf = value
        
    # Functions to faciliate dict-like access to / setting of attributes
        
    def __setitem__(self, key, val):
        setattr(self, key, val)
    
    
    def __getitem__(self, key):
        val = getattr(self, key)
        return val
        
    # Other methods
    
    def plot(self,i=None,j=None,
             positive_f_only:bool=True,
             axarr=None,
             **kwargs):
        """
        Function to plot frequency response matrix (f,G_f)
        """
        
        # Define rows/columns to plot
        if i is None:
            i = range(self.G_f.shape[1])
        if j is None:
            j = range(self.G_f.shape[2])
        
        # Create plots
        if axarr is None:
            fig, axarr = plt.subplots(len(i),len(j),sharex=True,sharey='row')
            
        else:
            fig = axarr[0,0].get_figure()
            
        fig.set_size_inches((14,8))
        
        fig.subplots_adjust(hspace=0.0,wspace=0.0)
        
        fig.suptitle("Plot of G(f) frequency response matrix")
        
        for row, _i in enumerate(i):
            
            for col, _j in enumerate(j):
                
                try:
                    ax = axarr[row,col]
                except IndexError:
                    break
                
                self.plot_component(_i,_j,
                                    ax_magnitude=ax,
                                    plotPhase=False,
                                    **kwargs)
                
                ax.set_ylabel(ax.get_ylabel(),
                              fontsize='x-small',
                              rotation=0,
                              horizontalAlignment='right',
                              wrap=True)
            
                # Tidy-up plot, removing labels etc.
                if col==0:
                    ax.set_ylabel(self.output_names[_i])
                else:
                    ax.set_ylabel("")
                    
                if row==0:
                    ax.set_title(self.input_names[_j])
                    
                if row!=len(i)-1:
                    ax.set_xlabel("")
                else:
                    if col!=0:
                        ax.set_xlabel("")
                        
            fig.subplots_adjust(left=0.20)
            fig.align_ylabels()
                    
        return fig, axarr
                                    
        
    def plot_component(self,i:int,j:int,
                       positive_f_only:bool=True,
                       label_str:str=None,
                       plotMagnitude:bool=True,ax_magnitude=None,
                       plotPhase:bool=True,ax_phase=None,
                       f_d:list=None) -> dict:
        """
        Function to plot frequency response matrix (f,G_f)
        
        ***
        Required:
           
        * `i`, `j`; indices to denote component of frequency response matrix
          to be plotted
            
        ***
        Optional:
            
        Variables:
        
        * `label_str`, used to label series in plot legend. If provided, legend 
          will be produced.
          
        * `f_d`, damped natural frequencies, used as vertical lines overlay
            
        Boolean options:
          
        * `plotMagnitude`, _boolean_, indicates whether magnitude plot required
        
        * `plotPhase`, _boolean_, indicates whether phase plot required
        
        Axes objects:
        
        * `ax_magnitude`, axes to which magnitude plot should be drawn
            
        * `ax_phase`, axes to which phase plot should be drawn
        
        If both plots are requested, axes should normally be submitted to both 
        `ax_magnitude` and `ax_phase`. Failing this a new figure will be 
        produced.
        
        ***
        Returns:
            
        `dict` containing figure and axes objects
        
        """
        
        f = self.f
        G_f = self.Gf[:,i,j]
        
        # Check shapes consistent
            
        if f.shape[0] != G_f.shape[0]:
            raise ValueError("Error: shape of f and G_f different!\n" +
                             "f.shape: {0}\n".format(f.shape) +
                             "G_f.shape: {0}".format(G_f.shape))
        
        # Create new figure with subplots if insufficient axes passed
        if (plotMagnitude and ax_magnitude is None) or (plotPhase and ax_phase is None):
            
            # Define new figure
            if plotMagnitude and plotPhase:
                
                fig, axarr = plt.subplots(2,sharex=True)    
                ax_magnitude =  axarr[0]
                ax_phase =  axarr[1]
                
            else:
                
                fig, ax = plt.subplots(1)
                
                if plotMagnitude:
                    ax_magnitude = ax
                else:
                    ax_phase = ax
                    
            # Define figure properties
            fig.suptitle("Frequency response G(f)")
            fig.set_size_inches((14,8))
            
        else:
            
            fig = ax_magnitude.get_figure()
        
        # Set x limits
        fmax = npy.max(f)
        fmin = npy.min(f)
        if positive_f_only:
            fmin = 0
        
        # Prepare magnitude plot
        if plotMagnitude:
            ax = ax_magnitude
            ax.plot(f,npy.abs(G_f),label=label_str) 
            ax.set_xlim([fmin,fmax])
            ax.set_xlabel("f (Hz)")
            ax.set_ylabel("|G(f)|")
            #ax.set_ylim([0.0,ax.get_ylim()[1]])
            if label_str is not None: ax.legend()
        
        # Prepare phase plot
        if plotPhase:
            ax = ax_phase
            ax.plot(f,npy.angle(G_f),label=label_str)
            ax.set_xlim([fmin,fmax])
            ax.set_ylim([-npy.pi,+npy.pi]) # angles will always be in this range
            ax.set_xlabel("f (Hz)")
            ax.set_ylabel("Phase G(f) (rad)")
            if label_str is not None: ax.legend()
        
        # Overlay vertical lines to denote pole frequencies
    
        if f_d is not None:
            
            for _f_d in f_d:
        
                ax_magnitude.axvline(_f_d,linestyle="--")
                ax_phase.axvline(_f_d,linestyle="--")
                    
        
        # Return objects via dict
        d = {}
        d["fig"]=fig
        d["ax_magnitude"] = ax_magnitude
        d["ax_phase"] = ax_phase
        return d

