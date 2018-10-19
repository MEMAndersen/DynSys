# -*- coding: utf-8 -*-
"""
Classes and methods used to store, present and manipulate eigenproperties
"""

import numpy as npy
import matplotlib.pyplot as plt
from functools import wraps


class Eig_Results():
    
    def __init__(self,s,X,Y,normalise=True):
        
        self.s = s
        self.X = X
        self.Y = Y
        
        self._sort_eigenproperties()
        
        if normalise:
            self._normalise_eigenvectors()
    

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__,self.__dict__)
    
    # Functions to faciliate dict-like access to / setting of attributes
    
    def __setitem__(self, key, val):
        setattr(self, key, val)
    
    def __getitem__(self, key):
        val = getattr(self, key)
        return val
    
    #def keys(self):
    #    return list(self.__dict__.keys())
    
    """
    Attribute getter methods
    """
    
    @property
    def s(self):
        """
        Array of eigenvalues
        """
        return self._s
    
    @property
    @wraps(s)
    def eigenvalues(self):
        return self.s
    
    # -----------
    @property
    def X(self):
        """
        Matrix of right eigenvectors
        """
        return self._X
    
    @property
    @wraps(X)
    def right_eigenvectors(self):
        return self.X
    
    # -----------
    @property
    def Y(self):
        """
        Matrix of left-eigenvectors
        """
        return self._Y
    
    @property
    @wraps(Y)
    def left_eigenvectors(self):
        return self.X
    
    # -----------
    @property
    def nModes(self):
        """
        Integer number of modes
        """
        return self._nModes
    
    # -----------
    @property
    def dynsys_obj(self):
        """
        Instance of `DynSys()` class (or derived classes thereof) to which 
        eigenproperties held by this object relate
        """
        return self._dynsys_obj
    
    @property
    @wraps(dynsys_obj)
    def dynsys(self):
        return self.dynsys_obj
    
    """
    Attribute setter methods
    """
    
    @s.setter
    def s(self,value):
        
        # Convert float (or int) to list of length 1
        try:
            len(value)
        except:
            value = [value]
        
        # Convert to ndarray if not already
        if not isinstance(value,npy.ndarray):
            value = npy.array(value,dtype=complex)
            
        # Check array dimensions
        if value.ndim != 1:
            raise ValueError("1D array expected!")
        
        self._s = value
        self._nModes = value.shape[0]
        
    @X.setter
    def X(self,value):
        
        value = self._convert2matrix(value,'X')
        self._check_shape(value,'X')
        self._X = value
        
    @Y.setter
    def Y(self,value):
        
        value = self._convert2matrix(value,'Y')    
        self._check_shape(value,'Y')            
        self._Y = value
        
    @dynsys_obj.setter
    def dynsys_obj(self,obj):
        self._dynsys_obj = obj
        
    @dynsys.setter
    def dynsys(self,obj):  # alternative setter method
        self.dynsys_obj = obj
    
    """
    Functions to determine engineering eigenproperties 
    from complex-valued eigensolution
    (as previously implemented within 'RealEigenvalueProperties()' method)
    
    Note various alternative spellings / access methods provided
    """    
    
    @property
    def f_d(self):
        """
        Damped natural frequency
        """
        return npy.imag(self.s) / (2*npy.pi)
    
    @property
    @wraps(f_d)
    def fd(self):
        return self.f_d
    
    # ----------
    @property
    def eta(self):
        """
        Damping ratio
        """
        s = self.s
        return - npy.real(s) / npy.absolute(s)
    
    @property
    @wraps(eta)
    def damping_ratio(self):
        return self.eta
    
    # ----------
    @property
    def f_n_abs(self):
        """
        Undamped natural frequency as positive real number
        """
        return npy.absolute(self.s) / (2*npy.pi)
    
    @property
    @wraps(f_n_abs)
    def fn_abs(self):
        return self.f_n_abs
    
    # ----------
    @property
    def f_n(self):
        """
        Undamped natural frequency as signed real number
        """
        return npy.sign(self.f_d) * self.f_n_abs
    
    @property
    @wraps(f_n)
    def fn(self):
        return self.f_n
    
    """
    Type conversion / checker methods
    """
    
    def _convert2matrix(self,value,varname):
        
        if not isinstance(value,npy.matrix):
            
            value = npy.array(value)
            
            if value.ndim == 2:
                value = npy.asmatrix(value,dtype=complex)
            
            else:
                raise ValueError("Error: matrix type expected for '%s'"
                                 % varname)
                
        return value
    
    
    def _check_shape(self,value,varname):
        
        Nm = self.nModes

        if value.shape[1] != Nm:
            raise ValueError("'%s' has unexpected shape!\n" % varname + 
                             "{0}.shape = {1}".format(varname, value.shape))
            
    """
    Other methods
    """
    
    def _normalise_eigenvectors(self):
        
        X, Y = self.X, self.Y
        self.X, self.Y = normalise_eigenvectors(X,Y)
        
    def _sort_eigenproperties(self):
        # Sort eigenvalues into ascending order of f_n_abs
        
        i = npy.argsort(self.f_n_abs)
        self.s = self.s[i]
        self.X = self.X[:,i]
        self.Y = self.Y[:,i]
        
        
    def plot(self,axarr=None):
        
        if axarr is None:
            axarr = [None,None,None,None]
            
        ax = axarr[0]
        ax = self.plot_orthogonality_check(ax=ax)
        
        ax = axarr[1]
        ax = self.plot_eigenvalues(ax=ax,plotType=1)
        
        ax = axarr[2]
        ax = self.plot_eigenvalues(ax=ax,plotType=3)
        
        ax = axarr[3]
        ax = self.plot_eigenvalues(ax=ax,plotType=4)
            
            
    def plot_eigenvalues(self,ax=None,plotType=1):
        """
        Plots eigenvalues (assumed to be complex) on the complex plane
        ***
        
        Allowable values for `plotType`:
            
        * `1`: Eigenvalues plotted on complex plane
        
        * `2`: |f_n| values plotted against index
        
        * `3`: f_d values plotted against index
        
        * `4`: Undamped natural frequencies vs damping ratio
        
        """
                    
        if ax is None:
            fig, ax = plt.subplots()
            
        if plotType == 1:
            
            s = self.s
            ax.plot(npy.real(s),npy.imag(s),'.b')
            ax.set_title("Eigenvalues plotted on complex plane")
            ax.set_xlabel("Real component")
            ax.set_ylabel("Imag component")
            
        elif plotType == 2:
            
            f_n_abs = self.fn_abs
            ax.bar(range(len(f_n_abs)),f_n_abs)
            ax.set_xlabel("Mode index")
            ax.set_ylabel("Undamped natural frequency (Hz)")
            ax.set_title("Natural frequencies vs mode index")
            
        elif plotType == 3:
            
            fd = self.fd
            ax.bar(range(len(fd)),fd)
            ax.set_xlabel("Mode index")
            ax.set_ylabel("Damped natural frequency (Hz)")
            ax.set_title("Damped natural frequencies vs mode index")
            
        elif plotType == 4:
            
            ax.plot(self.f_n,self.eta,'.b')
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Damping ratio")
            ax.set_title("Pole frequency vs damping ratio plot")
            
        else:
            raise ValueError("Error: unexpected plotType requested!")
            
        return ax
            
    
    def plot_orthogonality_check(self,ax=None):
        """
        Pixel plot of Y.T*X
        ***
        
        This can be used to check orthogonality of X and Y column vectors
        """
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
            
        # Produce pixel plot
        X = self.X
        Y = self.Y
        im = ax.imshow(npy.absolute(Y.T*X),interpolation='none',cmap='Greys')
        fig.colorbar(im)
        ax.set_title("Y.T * X product")
        
        return ax
        
    
# -------------- FUNCTIONS --------------
    
def normalise_eigenvectors(X,Y):
    """
    Normalise eigenvectors such that:
    
    $$ Y^{T}.X = I $$
    
    ***
    Required:
    
    * `X`, Numpy matrix, the columns of which are right-eigenvectors
    * `Y`, Numpy matrix, the columns of which are left-eigenvectors
    
    """
        
    d = npy.diagonal(Y.T * X)**0.5
    
    if d.any() != 0:
        X = X / d
        Y = Y / d
                    
    return X, Y
        


    
                   
        
        
if __name__ == "__main__":
    
    # Define some non-ideal inputs
    # note: will be converted to required type by setter methods
    s = [4.0+2j,-4+5j]
    X = npy.array([[1.0,3.0+6.0j],[-5.0,-8.0+6j]],dtype=complex)
    Y = [[6.0,2.0-4.0j],[2.5,-3.0+7j]]
    
    rslts = Eig_Results(s,X,Y)
    
    # Test __repr__ method
    print(rslts)
    
    # Try various methods for getting attributes
    s = rslts['eigenvalues']    # uses alt. getter method and __getitem__
    fn = rslts.fn               # conventional attribute access
    fd = rslts['fd']            # note tolerant to mis-spelling
    X = rslts.right_eigenvectors
    
    # Test setter function and dict __setitem__ behaviour
    rslts['s']=2.0
    print(rslts)
    
    # Test @wraps passes docstrings correctly
    help(Eig_Results.left_eigenvectors)
    help(Eig_Results.fn)
    help(Eig_Results.s)
    
    rslts['fd']=2.0 # should give AttributeError: can't set attribute
    


