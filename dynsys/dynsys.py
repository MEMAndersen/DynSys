# -*- coding: utf-8 -*-
"""
Classes used to define linear dynamic systems

@author: rihy
"""

from __init__ import __version__ as currentVersion

# Std library imports
import numpy as npy
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from pkg_resources import parse_version

import scipy.sparse as sparse

from scipy.linalg import block_diag
#from scipy.sparse import bmat

# DynSys module imports
from eig_results import Eig_Results
from freq_response_results import FreqResponse_Results
from common import convert2matrix, check_class
from mesh import Mesh


class DynSys:
    """
    Class used to store general properties and methods
    required to characterise a generic dynamic (2nd order) system
    """
    
    description="Generic dynamic system"

    def __init__(self,M,C,K,
                 J_dict=None,
                 output_mtrx=None,
                 output_names=None,
                 isLinear=True,
                 isModal=False,
                 isSparse=False,
                 name=None,
                 showMsgs=True,
                 mesh_obj=None):
        """
        Dynamic systems, which may have constraints, are defined by the 
        following:
            
        $$ M\ddot{y} + C\dot{y} + Ky = f $$
        $$ J\ddot{y} = 0 $$
        
        ***        
        Required arguments:
        
        * Mass matrix, `M`
            
        * Damping matrix, `C`
            
        * Stiffness matrix, `K`
        
        All must be square and of shape _[nxn]_ 
        
        ***
        Optional arguments:
        
        * `isLinear` (True/False required)
            
        * `J_dict`, _dict_ of constraint equations matrices. Shape of each 
        entry must be _[mxn]_
        
        * `mesh_obj`, instance of `Mesh` class. Allows mesh to be associated 
          with system
          
          _This is obviously useful for visualisation of results, but also 
          facilates certain analyses, e.g. integration of loads on a 
          system represented by modal properties_
            
        """
        
        # Convert to numpy matrix format
        M = npy.asmatrix(M)
        C = npy.asmatrix(C)
        K = npy.asmatrix(K)
        nDOF = M.shape[0]
        
        if not J_dict is None:
            for J in J_dict.values():
                J = npy.asmatrix(J)
                
        if isSparse:
            
            # Convert to scipy.sparse csr matrix format
            M = sparse.csc_matrix(M)
            C = sparse.csc_matrix(C)
            K = sparse.csc_matrix(K)
        
            if not J_dict is None:
                for J in J_dict.values():
                    J = sparse.csc_matrix(J)
            
        # Store as attributes
        self.DynSys_list=[self]
        """
        List of appended dynSys objects
        ***
        The purpose of this is to allow dynamic systems to be defined as an
        ensemble of `dynSys` class instances (or derived classes)
        
        Note by default `self` is always included in this list
        """
        
        self._M_mtrx = M
        """Mass matrix"""
        
        self._C_mtrx = C
        """Damping matrix"""
        
        self._K_mtrx = K
        """Stiffness matrix"""
        
        self.nDOF = nDOF
        """Number of degrees of freedom"""
        
        self.isLinear = isLinear
        """Boolean, describes whether system is linear or not"""
        
        self.isModal = isModal
        """
        Used to denote that dofs are _modal_, rather than real-world freedoms
        """
        
        self.isSparse = isSparse
        """
        Boolean, denotes whether system matrices should be stored and 
        manipulated as sparse matrices
        """
        
        self._J_dict = {}
        """Dict of constraints matrices"""
        
        if J_dict is not None:
            self._J_dict = J_dict
        
        if name is None:
            name = self.__class__.__name__
        self.name = name
        """Name/descriptions of dynamic system"""
        
        if output_mtrx is None:
            output_mtrx = []
        self.output_mtrx = output_mtrx
        
        if output_names is None:
            output_names = []
        self.output_names = output_names
                            
        self.mesh = mesh_obj
        
        # Check definitions are consistent
        self._CheckSystemMatrices()
        self.check_outputs()
        
        if showMsgs:
            print("%s `%s` initialised." % (self.description,self.name))
        
        if isSparse:
            print("Note: sparse matrix functionality as provided by Scipy "
                  "will be used for system matrices")
            
    # ------------------ GETTER / SETTER METHODS -----------------
    
    @property
    def output_mtrx(self):
        """
        List of output matrices
        
        Use `add_outputs()` to define append output matrices. 
        `check_outputs()` can be used to check the validity (shape) of the 
        output matrices defined.
        """
        return self._output_mtrx_list
    
    @output_mtrx.setter
    def output_mtrx(self,value):
        if value!=[]:
            value = convert2matrix(value)
            value = [value]
        self._output_mtrx_list = value
        
        
    def has_output_mtrx(self)->bool:
        """
        Returns True if system has output matrix
        """
        if self.output_mtrx == []:
            return False
        else:
            return True
        

    # --------------   
    @property
    def output_names(self):
        """
        List of string descriptions for rows of `output_mtrx`.
        Used to label plots etc.
        """
        return self._output_names_list
        
    @output_names.setter
    def output_names(self,value):
        if value!=[]:
            value = list(value) # convert to list
            value = [value]     # make nested list
        self._output_names_list = value
        
        
    # --------------   
    @property
    def mesh(self):
        """
        Returns instance of `Mesh` class, used to define mesh that relates to 
        system
        """
        return self._mesh
    
    @mesh.setter
    def mesh(self,obj):
        
        check_class(obj,Mesh)
        self._mesh = obj
        
    
    def has_mesh(self)->bool:
        """
        Returns True if object has `Mesh` instance associated with it
        """
        if self.mesh is None:
            return False
        else:
            return True
        
    # --------------                 
        
    def _CheckSystemMatrices(self,
                             nDOF=None,
                             M_mtrx=None,
                             C_mtrx=None,
                             K_mtrx=None,
                             checkConstraints=True,
                             J_dict=None):
        """
        Function carries out shape checks on system matrices held as class 
        attributes
        
        Matrices held as object attributes will be used, unless optional 
        arguments are provided
        """

        # Handle optional arguments
        if nDOF is None:    nDOF = self.nDOF
        if M_mtrx is None:  M_mtrx = self._M_mtrx
        if C_mtrx is None:  C_mtrx = self._C_mtrx
        if K_mtrx is None:  K_mtrx = self._K_mtrx
        if J_dict is None:  J_dict = self._J_dict

        # Check shapes consistent with stated nDOF
        if C_mtrx.shape[0]!=nDOF:
            raise ValueError("Error: C matrix row dimension inconsistent!\n"
                             + "Shape: {0}".format(C_mtrx))
        if K_mtrx.shape[0]!=nDOF:
            raise ValueError("Error: K matrix row dimension inconsistent!\n"
                             + "Shape: {0}".format(K_mtrx))
        
        # Check matrices are square
        if M_mtrx.shape[1]!=nDOF:
            raise ValueError("Error: M matrix not square!\n"
                             + "Shape: {0}".format(M_mtrx))
        if C_mtrx.shape[1]!=nDOF:
            raise ValueError("Error: C matrix not square!\n"
                             + "Shape: {0}".format(C_mtrx))
        if K_mtrx.shape[1]!=nDOF:
            raise ValueError("Error: K matrix not square!\n"
                             + "Shape: {0}".format(K_mtrx))
            
        # Check shape of all constraints matrices
        if checkConstraints:
            for key, J_mtrx in J_dict.items():
                
                if J_mtrx.shape[1]!=nDOF:    
                    raise ValueError("Error: J matrix column dimension " + 
                                     "inconsistent!\n"
                                     + "Shape: {0}\n".format(J_mtrx.shape)  
                                     + "J_mtrx: {0}".format(J_mtrx))
            
        return True
    
        
    def check_outputs(self,output_mtrx=None,output_names=None,verbose=False):
        """
        Checks that all defined output matrices are of the correct shape
        """
        
        # Handle option arguments
        if output_mtrx is None:
            output_mtrx = self.output_mtrx
            
        if output_names is None:
            output_names = self.output_names
            
        # Exit early if both none
        if output_mtrx is None and output_names is None:
            return True
        
        if verbose:
            
            print("\nCheckOutputMtrx() method invoked:")
        
            print("Output matrix shapes:")
            for _om in output_mtrx:
                print(_om.shape)
            
            print("Output names:")
            print(output_names)
        
        # Check list lengths agree
        for _om, _names in zip(output_mtrx, output_names):
            
            if len(_names)!=_om.shape[0]:
                raise ValueError("Length of lists `output_names` "+
                                 "and rows of `output_mtrx` do not agree!\n"+
                                 "len(output_names)={0}\n".format(len(_names))+
                                 "output_mtrx.shape: {0}".format(_om.shape))
        
            # Check shape of output matrix 
            nDOF_expected = 3*self.nDOF
            
            if _om.shape[1] != nDOF_expected:
                raise ValueError("output_mtrx of invalid shape defined!\n" +
                                 "Shape provided: {0}\n".format(_om.shape) +
                                 "Cols expected: {0}".format(nDOF_expected))
    
        return True
    
    
    def ReadOutputMtrxFromFile(self,
                                fName='outputs.csv'):
        """
        Reads output matrix file. Output matrix format required is as follows:
        ***
        
        $$ y = C.x $$
        
        where:
        $$ x = [q,\dot{q},\ddot{q}]^{T} $$ 
        is the (extended) _state variable_ vector_ and y is _output vector_. 
        C is the _output matrix_ mapping _state variables_ **x** to _outputs_ **y**.
        
        """
        
        df = pd.read_csv(fName,delimiter=',',header=0,index_col=0)
    
        C_mtrx = npy.asmatrix(df)
        outputNames = df.index.tolist()
        
        return C_mtrx,outputNames 
    
    
    def AddOutputMtrx(self,*args,**kwargs):
        self.add_outputs(*args,**kwargs)
        
    
    def add_outputs(self,output_mtrx=None,output_names=None,
                    fName='outputs.csv',
                    append=True,verbose=False):
        """
        Appends new output matrix and associated names to object
        ***
        
        Optional:
            
        * `output_mtrx`, numpy matrix expected
        
        * `output_names`, list or array of strings
        
        * `fName`, string denoting csv file defining output matrix and names
        
        * `append`, if True (default) then new output matrices and names will 
          be appended to any previously-defined outputs.
        
        For normal usage either `output_mtrx` and `output_names` to be 
        provided. Otherwise an attempt will be made to read data from `fName`.
        """
        
        if verbose:
            print("'add_outputs()' method invoked.")
        
        # Read from file if no output_mtrx provided
        if output_mtrx is None:
            if verbose:
                print("New outputs defined in '%s'" % fName)
            output_mtrx, output_names = self.ReadOutputMtrxFromFile(fName)
            
        # Create default output names if none provided
        if output_names is None:
            output_names = ["Response {0}".format(x)
                            for x in range(output_mtrx.shape[0])]
                    
        if append:
            
            output_mtrx = convert2matrix(output_mtrx)
            output_names = list(output_names)
            
            self.output_mtrx.append(output_mtrx)
            self.output_names.append(output_names)
            
        else:
            
            self.output_mtrx = output_mtrx
            self.output_names = output_names            
        
        if verbose:
            
            print("Updated output matrix shapes:")
            for _om in self.output_mtrx:
                print(_om.shape)
            
            print("Updated output names:")
            print(self.output_names)
        
        # Check dimensions of all output matrices defined
        self.check_outputs()
        
    
    def PrintSystemMatrices(self,printShapes=True,printValues=False):
        """
        Function is used to print system matrices to text window   
        ***
        Useful for documentation and debugging
        """
        
        print("**** PrintSystemMatrices() : `{0}` ****\n".format(self.name))
        
        # Print names of all systems and sub-systems
        names_list = [x.name for x in self.DynSys_list]
        print("Systems list:")
        print(names_list)
        print("")
        
        # Loop through all systems and subsystems
        for x in self.DynSys_list:
                
            print("---- System matrices for `{0}` ----\n".format(x.name))
            
            # Print general system matrices
            attr_list = ["_M_mtrx", "_C_mtrx", "_K_mtrx"]
            
            for attr in attr_list:
            
                if hasattr(x,attr):
                    
                    val = getattr(x,attr)
                    
                    print("{0} matrix:".format(attr))
                    print(type(val))
                    
                    if printShapes: print(val.shape)
                    if printValues: print(val)
                    
                    print("")
                    
            # Print constraints matrices
            print("---- Constraint matrices for `{0}` ----\n".format(x.name))
            
            if not x._J_dict:
                print("(No constraints matrices defined)\n")
                
            else:
                
                for key, val in x._J_dict.items():
                    
                    print("key: {0}".format(key))
                    print(type(val))
                    
                    if printShapes: print(val.shape)
                    if printValues: print(val)
                    
                    print("")
    

    def GetSystemMatrices(self,
                          unconstrained:bool=False,
                          createNewSystem:bool=False):
        """
        Function is used to retrieve system matrices, which are not usually to 
        be accessed directly, except by member functions
        
        ***
        Optional:
            
        * `unconstrained`, boolean, if True system matrices applicable to 
          the _unconstrained problem_ are returned. Note: only applicable to 
          systems with constraint equations. Default value = False.
          Refer documentation for `transform_to_unconstrained()` for details.
          
        * `createNewSystem`, boolean, if True a new `DynSys()` class instance 
          is initialised, using the system matrices of the full system.
          Default value = False.
            
        ***
        Returns:
            
        Matrices (and other results) are returned as a dictionary
        """
        
        # Create empty dictionay
        d = {}
        
        # Get list of systems
        DynSys_list = self.DynSys_list
        
        # Determine properties of overall system
        isLinear = all([x.isLinear for x in DynSys_list])
        isSparse = all([x.isSparse for x in DynSys_list])
        
        # Retrieve system matrices from all listed systems
        nDOF_list = []
        M_list = []
        C_list = []
        K_list = []
        J_key_list = []
        
        for x in DynSys_list:
            
            nDOF_list.append(x.nDOF)
            
            M_list.append(x._M_mtrx.tolist())
            C_list.append(x._C_mtrx.tolist())
            K_list.append(x._K_mtrx.tolist())

            # Compile list of all J keys
            J_key_list += list(x._J_dict.keys())
            
        J_key_list = list(set(J_key_list)) # remove duplicates
            
        # Assemble system matrices for full system 
        # i.e. including all appended systems
        nDOF_new = sum(nDOF_list)
        M_mtrx = block_diag(*tuple(M_list))
        C_mtrx = block_diag(*tuple(C_list))
        K_mtrx = block_diag(*tuple(K_list))
        
        # Assemble constraints matrix for full system
        J_dict = {}
        
        for key in J_key_list:
            
            J_list = []
            m=0 # denotes number of constraint equations
            
            for x in DynSys_list:
                
                if key in list(x._J_dict.keys()):
                    
                    J_mtrx = x._J_dict[key]
                    m = J_mtrx.shape[0]
                    
                    if x.isSparse:
                        J_mtrx = sparse.csc_matrix(J_mtrx)
                    
                    J_list.append(J_mtrx)
                    
                else:
                    
                    J_list.append(npy.asmatrix(npy.zeros((m,x.nDOF))))
                
            # Assemble rows of full matrix
            full_J_mtrx = npy.hstack(tuple(J_list))            
            J_dict[key] =full_J_mtrx
            
        # Assemble full constraints matrix
        if J_dict:
            J_mtrx = npy.vstack(list(J_dict.values()))
        else:
            J_mtrx = npy.zeros((0,nDOF_new))
        
        # Check shapes of new matrices
        self._CheckSystemMatrices(nDOF=nDOF_new,
                                  M_mtrx=M_mtrx,
                                  C_mtrx=C_mtrx,
                                  K_mtrx=K_mtrx,
                                  J_dict=J_dict)
        
        self.CheckConstraints(J=J_mtrx,verbose=False)
        
        # Project system matrices onto null space of constraints matrix 
        # to transform to unconstrained problem
        if unconstrained and self.hasConstraints():
            
            mdict = transform_to_unconstrained(J=J_mtrx,M=M_mtrx,
                                               C=C_mtrx,K=K_mtrx)
            M_mtrx = mdict["M"]
            C_mtrx = mdict["C"]
            K_mtrx = mdict["K"]
            Z_mtrx = mdict["Null_J"]
        
        # Populate dictionary
        d["nDOF"] = nDOF_new
        
        d["M_mtrx"]=M_mtrx
        d["C_mtrx"]=C_mtrx
        d["K_mtrx"]=K_mtrx
        
        d["J_dict"]=J_dict        
        d["J_mtrx"]=J_mtrx
        
        d["isLinear"]=isLinear
        d["isSparse"]=isSparse
        
        if unconstrained and self.hasConstraints():
            d["Null_J"]=Z_mtrx
        
        # Create new system object, given system matrices
        if createNewSystem:
            
            DynSys_full = DynSys(M=M_mtrx,
                                 C=C_mtrx,
                                 K=K_mtrx,
                                 J_dict=J_dict,
                                 isLinear=isLinear,
                                 isSparse=isSparse,
                                 name=[x.name for x in self.DynSys_list],
                                 showMsgs=False)
            
            d["DynSys_full"]=DynSys_full
        
        # Return dictionary
        return d
    
    def AddConstraintEqns(self,Jnew,Jkey,checkConstraints=True):
        """
        Function is used to append a constraint equation
        ***
        
        Constraint equations are assumed to take the following form:
        $$ J\ddot{y} = 0 $$
        
        ***
        Required:
        
        * `Jnew`, _matrix_ of dimensions _[m,n]_ where:    
        
            * _m_ denotes the number of constraint equations
            
            * _n_ denotes the number of DOFS of the system
            
        * `Jkey`, key used to denote Jnew within dict    
            
        ***
            
        **Important note**: `Jnew` must be *full rank*, i.e. must itself have
        independent constraints. In addition `Jnew` must be independent of any 
        constraints previously defined.
        
        `CheckConstraints()` should be used to test whether constraint 
        equations are independent
        """
        
        # Convert Jnew to appropriate representation
        if not self.isSparse:
            Jnew = npy.asmatrix(Jnew)
        else:
            Jnew = sparse.csc_matrix(Jnew)
        
        # Check dimensions
        if Jnew.shape[1]!=self.nDOF:
            raise ValueError("Error: constraint eqn dimensions inconsistent!")
                
        # Store constraint equation as new dict item
        self._J_dict[Jkey]=Jnew
            
        # Check constraints matrix is valid
        if checkConstraints:
            self.CheckConstraints()
            
            
    def CalcStateMatrix(self,
                        M=None,
                        C=None,
                        K=None,
                        nDOF=None,
                        unconstrained=False,
                        saveAsAttr:bool=True):
        """
        Assembles the continous-time state matrix `A_mtrx` used in state-space 
        methods
        ***
        
        The continuous-time state matrix is as follows:
            
        $$ A = [[0,I],[-M^{-1}K,-M^{-1}C]] $$
        
        where **M** is the system mass matrix, **C** is the system damping 
        matrix, **K** is the system stiffness matrix and **I** is an 
        identity matrix.
        
        ***
        Optional:
            
        _Unless optional arguments are specified, system matrices stored as 
        class attributes will be used._
            
        * `M`, mass matrix
        
        * `C`, damping matrix
        
        * `K`, stiffness matrix
        
        * `unconstrained`, _boolean_; if True load matrix for the 
          _unconstrained problem_ will be returned. Only applicable if 
          constraint equations are defined. Refer documentation of 
          `transform_to_unconstrained()` method for further details
                    
        * `saveAsAttr`: if `True` state matrix returned will also be saved as 
          an object instance attribute
        
        """
        
        # Retrieve system matrices
        d = self.GetSystemMatrices(unconstrained=unconstrained)
        
        # Handle optional arguments
        if M is None: M = d["M_mtrx"]
        if C is None: C = d["C_mtrx"]
        if K is None: K = d["K_mtrx"]
        if nDOF is None: nDOF = d["nDOF"]
    
        # Check shape of system matrices
        self._CheckSystemMatrices(M_mtrx=M,
                                  C_mtrx=C,
                                  K_mtrx=K,
                                  checkConstraints=False,
                                  nDOF=M.shape[0])
        
        # Assemble state matrix
        A, Minv = calc_state_matrix(M=M,K=K,C=C,isSparse=self.isSparse)
        
        # Save as attribute
        if saveAsAttr:
            self._A_mtrx = A
            self._Minv = Minv
            
            if unconstrained:
                self._Null_J = d["Null_J"]
        
        return A     
    
    
    def GetStateMatrix(self,
                       unconstrained=False,
                       recalculate=True):
        """
        Helper function to obtain state matrix, if already calculated 
        and held as attribute. Otherwise state matrix will be recalculated
        
        ***
        Optional:
            
        * `unconstrained`, _boolean_; if True load matrix for the 
          _unconstrained problem_ will be returned. Only applicable if 
          constraint equations are defined.
          
        * `recalculate`, _boolean_; if True load matrix will always be 
          re-evaluated upon function call. Otherwise if load matrix has already 
          been evaluated for system (and is held as attribute) then it will 
          not be re-evaluated.
        
        """
        
        attr = "_A_mtrx"
        
        if recalculate or not hasattr(self,attr):
            return self.CalcStateMatrix(unconstrained=unconstrained)
        else:
            return getattr(self,attr)
    
    
    def CalcLoadMatrix(self,
                       M=None,
                       unconstrained=False,
                       saveAsAttr=True):
        """
        Assembles the load matrix `B_mtrx` used in state-space methods
        ***
        
        Load matrix **B** is given by the following:
            
        $$ B = [[0],[M^{-1}]] $$
        
        where **M** is the system mass matrix.
        
        ***
        Optional:
            
        _Unless optional arguments are specified, system matrices stored as 
        class attributes will be used._
            
        * `M`, mass matrix
        
        * `unconstrained`, _boolean_; if True load matrix for the 
          _unconstrained problem_ will be returned. Only applicable if 
          constraint equations are defined. Refer documentation of 
          `transform_to_unconstrained()` method for further details
        
        * `saveAsAttr`: if `True` state matrix returned will also be saved as 
          an object instance attribute
        
        """
        
        hasConstraints = self.hasConstraints()
        
        # Retrieve system matrices
        if M is None:
            
            mdict = self.GetSystemMatrices(unconstrained=unconstrained)
            M = mdict["M_mtrx"]
            
            if hasConstraints and unconstrained:
                J = mdict["J_mtrx"]
                Z = mdict["Null_J"]
        
        else:
            self._CheckSystemMatrices(M_mtrx=M)
        
        # Convert to unconstrained problem, if applicable
        if unconstrained and hasConstraints:

            Minv = npy.linalg.inv(M)
            Minv = Minv @ Z.T
            B, Minv = calc_load_matrix(M=None,Minv=Minv,isSparse=False)
            
        else:
            B, Minv = calc_load_matrix(M=M,isSparse=self.isSparse)
            
        # Check shape
        nDOF = mdict["nDOF"]
        
        if B.shape[1]!=nDOF:
            raise ValueError("Unexpected column dimension for 'B' matrix!")
        
        if unconstrained and hasConstraints:
            if B.shape[0]!=2*(nDOF-J.shape[0]):
                raise ValueError("Unexpected row dimension for 'B' matrix "+
                                 "applicable to unconstrained problem")
        
        else:
            if B.shape[0]!=2*nDOF:
                raise ValueError("Unexpected row dimension for 'B' matrix")
        
        # Save as attribute
        if saveAsAttr:
            self._B_mtrx = B
            self._Minv = Minv
            
        return B
    
    
    def GetLoadMatrix(self,
                      unconstrained:bool=False,
                      recalculate:bool=False):
        """
        Helper function to obtain load matrix, if already calculated 
        and held as attribute. Otherwise load matrix will be recalculated
        
        ***
        Optional:
            
        * `unconstrained`, _boolean_; if True load matrix for the _unconstrained 
          problem_ will be returned. Only applicable if constraint equations 
          are defined.
          
        * `recalculate`, _boolean_; if True load matrix will always be 
          re-evaluated upon function call. Otherwise if load matrix has already 
          been evaluated for system (and is held as attribute) then it will 
          not be re-evaluated.
            
        """
        
        attr = "_B_mtrx"
        
        if recalculate or not hasattr(self,attr):
            return self.CalcLoadMatrix(unconstrained=unconstrained)
        else:
            return getattr(self,attr)
        
        
    def get_output_mtrx(self,
                        state_variables_only:bool=False,
                        all_systems:bool=True):
        """
        Returns output matrix for overall system
        
        ***
        Optional:
            
        * `state_variables_only`, _boolean_, if True, only columns relating to 
          state variables (i.e. displacements, velocities - but not 
          accelerations) will be returned
          
        * `all_systems`, _boolean_, if True output matrices for all subsystems 
          will be arranged as block diagonal matrix, which represents the 
          output matrix for the full system
          
        """
        
        # Define list over which to loop
        if all_systems:
            sys_list = self.DynSys_list
        else:
            sys_list = [self]
        
        # Assemble full output matrix by arranging as block diagonal matrix
        
        disp_cols_list = []
        vel_cols_list = []
        accn_cols_list = []
        output_names_list = []
        
        for x in sys_list:

            nDOF = x.nDOF
                        
            # Loop over all output matrices
            for i, (om, names) in enumerate(zip(x.output_mtrx,x.output_names)):
                                
                if i==0:
                    output_mtrx = om
                    output_names = names
                    
                else:
                    output_mtrx = npy.vstack((output_mtrx,om))
                    output_names = output_names + names
                                
            # Decompose into groups relating to (disp,vel,accn)
            disp_cols = output_mtrx[:,:nDOF]
            vel_cols = output_mtrx[:,nDOF:2*nDOF]
            accn_cols = output_mtrx[:,2*nDOF:]
            
            # Append to lists
            disp_cols_list.append(disp_cols)
            vel_cols_list.append(vel_cols)
            accn_cols_list.append(accn_cols)
            output_names_list.append([x.name+" : "+y for y in output_names])
            
        # Break out of function if no output matrices defined
        if output_names_list==[]:
            return None, None
            
        # Assemble submatrices for full system
        disp_cols_full = scipy.linalg.block_diag(*disp_cols_list)
        vel_cols_full = scipy.linalg.block_diag(*vel_cols_list)
        
        if not state_variables_only:
            accn_cols_full = scipy.linalg.block_diag(*accn_cols_list)
        
        # Concatenate to prepare output matrix for full system
        output_mtrx_full = npy.hstack((disp_cols_full,vel_cols_full))
        
        if not state_variables_only:
            output_mtrx_full = npy.hstack((output_mtrx_full,accn_cols_full))
        
        # Convert list of names to array format
        output_names_arr = npy.ravel(output_names_list) 
        
        # Return matrix and row names for full system
        return output_mtrx_full, output_names_arr
    
    
    def EqnOfMotion(self,x, t,
                    forceFunc,
                    M,C,K,J,
                    nDOF,
                    isSparse,
                    isLinear,
                    hasConstraints):
        """
        Function defines equation of motion for dynamic system
        ***
        
        The behaviour of 2nd order dynamic systems is characterised by the 
        following ODE:
            
        $$ M\ddot{y} + C\dot{y} + Ky = f $$
        
        This can be re-arranged as:
            
        $$ \ddot{y} = M^{-1}(f - C\dot{y} - Ky) $$
        
        Dynamic systems may have constraint equations defined as follows:
            
        $$ J\ddot{y} = 0 $$
            
        """
        
        isDense = not isSparse
        
        # Check system is linear
        if not isLinear:
            raise ValueError("System `{0}` is not linear!".format(self.name))
        
        # Obtain inverse mass matrix
        attr = "_M_inv"
        
        if hasattr(self,attr):
        
            Minv = getattr(self,attr)
        
        else:
            
            if isDense:
                Minv = npy.linalg.inv(M)
            else:
                Minv = sparse.linalg.inv(M)
            
            
            setattr(self,attr,Minv)
        
        if hasConstraints:
                
            # Obtain inverse of J.Minv.J.T
            attr1 = "_A_inv"
            
            if hasattr(self,attr1):
            
                Ainv = getattr(self,attr1)
                
            else:
                
                # Multiply matrices
                A = J @ Minv @ J.T
                
                if isDense:
                    Ainv = npy.linalg.inv(A)
                else:
                    Ainv = sparse.linalg.inv(A)
                    
                setattr(self,attr1,Ainv)
        
        # Convert to column vector format
        x = npy.asmatrix(x).T
        
        # Split x into components
        y = x[:nDOF]
        ydot = x[nDOF:]
        
        # Get input force at time t
        f = npy.asmatrix(forceFunc(t)).T
        
        # Calculate net force (excluding constraint forces
        f_net = f - K.dot(y) - C.dot(ydot)
        
        # Solve for accelerations
        if hasConstraints:
            
            # Calculate lagrange multipliers (to satify constraint eqns)
            lagrange = Ainv * (- J * Minv * f_net)
            
            # Define acceleration
            y2dot = Minv*(J.T*lagrange + f_net)
        
        else:
            
            lagrange = npy.asmatrix(npy.zeros((0,1)))
                
            y2dot = Minv*f_net
            
        # Obtain constraint forces
        f_constraint = J.T * lagrange
        
        # Returns results as dict
        d = {}
        d["t"]=t
        d["y2dot"]=y2dot
        d["ydot"]=ydot
        d["y"]=y
        d["f"]=f
        d["lagrange"]=lagrange
        d["f_constraint"]=f_constraint
        return d
    
        
    def CalcEigenproperties(self,*args,**kwargs):
        """
        Deprecated method name. 
        Refer docstring for `calc_eigenproperties()` method
        """
        return self.calc_eigenproperties(*args,**kwargs)
    
    
    def calc_eigenproperties(self,
                             normalise=True,
                             verbose=False,
                             makePlots=False,
                             axarr=None):
        """
        General method for determining damped eigenvectors and eigenvalues 
        of system
        ***
        
        Note in general eigenproperties will be complex due to non-proportional
        damping.
        
        Eigendecomposition of the system state matrix 'A' is carried out to 
        obtain eigenvalues and displacement-velocity eigenvectors.
        
        Engineers who are not familiar with the background theory should read
        the following excellent paper:
            
        *An Engineering Interpretation of the Complex 
        Eigensolution of Linear Dynamic Systems*
        
        by Christopher Hoen. 
        
        [PDF](../references/An Engineering Interpretation of the Complex 
        Eigensolution of Linear Dynamic Systems.pdf)
        
        ***
        **Required:**
            
        No arguments; the mass, stiffness, damping and (if defined) constraint 
        matrices held as attributes of the system will be used.
        
        ***
        **Optional:**
            
        * `normalise`, _boolean_, dictates whether eigenvectors 
          should be normalised, such that Y.T @ X = I
          
        * `makePlots`, _boolean_, if True plots will be produced to illustrate 
          the eigensolution obtained
          
        * `axarr`, list of _axes_ onto which plots should be made. If None 
          plots will be made onto new figures
          
        * `verbose`, _boolean_, if True intermediate output & text will be 
          printed to the console
         
        ***
        **Returns:**
             
        Instance of `Eig_Results` class
                
        """
        
        # Get system matrices
        d = self.GetSystemMatrices()
        M = d["M_mtrx"]
        K = d["K_mtrx"]
        C = d["C_mtrx"]
        
        if self.hasConstraints():
            J = d["J_mtrx"]
        else:
            J = None
            
        # Compute eigenproperties of A_c
        # s is vector of singular values
        # columns of X are right eigenvectors of A
        # columns of Y are left eigenvectors of A  
        eig_rslts_obj = solve_eig(M=M,K=K,C=C,J=J,
                                  normalise=normalise,
                                  verbose=verbose)
        
        # Create two-way link between objects
        eig_rslts_obj.dynsys = self
        self.eig_rslts = eig_rslts_obj
        
        if makePlots:
            eig_rslts_obj.plot(axarr)
                    
        return eig_rslts_obj 
    
    
    def CheckDOF(self,DOF):
        """
        Function is used to check is a certain requested DOF index is valid
        """
                
        if hasattr(self,"isModal") and self.isModal:
            # Modal systems
            
            if DOF < 0 or DOF >= self.nDOF_realWorld:
                raise ValueError("Error: requested real-world DOF invalid!")
        
        else:
            # Non-modal systems, i.e. DOFs are real-world
            
            if DOF < 0 or DOF >= self.nDOF:
                raise ValueError("Error: requested DOF invalid!")
        
        return True
    
    
    def hasConstraints(self)->bool:
        """
        Tests whether constraint equations are defined
        """
        
        if len(self._J_dict)>0:
            return True
        else:
            return False
    
    
    def CheckConstraints(self,J=None,verbose=True,raiseException=True)->bool:
        """
        Check contraint equations are independent
        ***
        Practically this is done by checking that the full `J_mtrx` of the 
        system, including any sub-systems, is full rank
        """

        if J is None:
            d = self.GetSystemMatrices(createNewSystem=True)
            J = d["J_mtrx"]
            full_sys = d["DynSys_full"]
        else:
            full_sys = self
            
        if J.shape[0]==0:
            return True # no constraint equations defined
        
        if verbose:
            print("Checking constraint equations for `%s` " % full_sys.name)
        
        m = J.shape[0]
        if verbose:
            print("Number of constraint equations: %d" % m)
            
        if self.isSparse:
            J = J.todense()
        
        r = npy.linalg.matrix_rank(J)
        if verbose:
            print("Number of independent constraint equations: %d" % r)
    
        if m!=r:
            
            errorStr="Error: constraints matrix not full rank!\n"
            errorStr+="J.shape: {0}\nComputed rank: {1}".format(J.shape,r)
            
            if raiseException:
                raise ValueError(errorStr)
            else:
                print(errorStr) # do not raise exception - but print to console
                
            return False
        
        else:
            if verbose: print("Constraints are independent, as required")
                    
            return True
    
    
    def AppendSystem(self,
                     child_sys,
                     J_key:str=None,
                     
                     Xpos_parent:float=None,
                     modeshapes_parent=None,
                     DOF_parent:int=None,
                     
                     Xpos_child:float=None,
                     modeshapes_child=None,
                     DOF_child:int=None,
                     ):
        """
        Function is used to join two dynamic systems by establishing 
        appropriate constraint equations
        
        ***
        Required arguments:
            
        * `child_sys`, _ DynSys_ instance describing child system, i.e. system 
          to be appended
          
        ***
        Optional arguments:
            
        * `J_key`, _string_ identifier used in constraints dict. If _None_ then 
          default key will be established
            
        Usage of optional arguments depends on the properties of the parent 
        and child systems, as follows:
            
        **Parent system:**
            
        * If _isModal_:
            
            * `Xpos_parent` can be used to define the point on the _parent 
              system_ at which the child system is to be attached. 
              Note: usage of this parameter requires the _parent system_ to 
              have function attribute `modeshapeFunc`, i.e. a function 
              describing how modeshapes vary with chainage.
              
            * Alternatively `modeshapes_parent` can be used to directly 
              provide modeshape vector relevant to the point on the _parent 
              system_ at which the child system is to be attached
              
            If both are provided, `modeshapes_parent` is used to define 
            modeshapes, i.e. take precedence.
            
        * Else:
            
            * `DOF_parent` should be used to specify the index of the DOF in 
              the _parent system_ to which the child system is to be attached
              
        **Child system:**
        
        Similar logic applies as per parent systems:
        
        * If _isModal_:
            
            * `Xpos_child` can be used to define the point on the _child 
              system_ at which the parent system is to be attached. 
              Note: usage of this parameter requires the _child system_ to have 
              function attribute `modeshapeFunc`, i.e. a function describing 
              how modeshapes vary with chainage.
              
            * Alternatively `modeshapes_child` can be used to directly 
              provide modeshape vector relevant to the point on the _child 
              system_ at which the parent system is to be attached
            
        * Else:
            
            * `DOF_child` should be used to specify the index of the DOF in 
              the _child system_ to which the parent system is to be attached
        
        **Note**: this function can only be used with *linear* systems 
        with constant `M`, `C`, `K` system matrices.
        (This is checked: an exception will be raised if attempt is made to use 
        with nonlinear systems).
        
        """
        
        parent_sys = self   # for clarity in the following code
        
        # Add child system to parent system's list
        parent_sys.DynSys_list.append(child_sys)
        
        # Define default key
        if J_key is None:
            J_key = "0_%d" % (len(parent_sys.DynSys_list)-1)
        
        def link_sys(sys_obj, sys_type:str, Xpos, modeshapes, DOF):
            """
            Function to carry out the necessary tasks to link either
            parent or child system
            """
            
            # Check system is linear
            if not sys_obj.isLinear:
                raise ValueError("{0} system '{1}' is not linear!"
                                 .format(sys_type,sys_obj.name))
        
            # Factor to apply to give equal and opposite behaviour
            if sys_type == "parent":
                factor = +1
            elif sys_type == "child":
                factor = -1
            else:
                raise ValueError("Unexpected `sys_type`!")
        
            # Logic as per docstring
            if sys_obj.isModal:
                
                if Xpos is not None:
                    
                    if modeshapes is None:
                        
                        attr = "modeshapeFunc"
                        
                        if hasattr(sys_obj,attr):
                            modeshapes = getattr(sys_obj,attr)(Xpos)
                        
                        else:
                            raise ValueError("`Xpos` argument provided but " + 
                                             "{0} system '{1}'"
                                             .format(sys_type,sys_obj.name) + 
                                             "does not have function " + 
                                             "attribute `%s'" % attr)    
                
                    # Save as attributes
                    attr = "Xpos_attachedSystems"
                    if hasattr(sys_obj,attr):
                        getattr(sys_obj,attr).append(Xpos)
                    else:
                        setattr(sys_obj,attr,[Xpos])
                        
                    attr = "modeshapes_attachedSystems"
                    if hasattr(sys_obj,attr):
                        getattr(sys_obj,attr).append(modeshapes)
                    else:
                        setattr(sys_obj,attr,[modeshapes])
                
                elif modeshapes is None:
                    raise ValueError("{0} system is modal. ".format(sys_type) + 
                                     "Either `Xpos_{0}` ".format(sys_type) + 
                                     "or `modeshapes_{0}` ".format(sys_type) + 
                                     "arguments are required")
                    
                # Define new constraint equation submatrix
                J_new = factor * modeshapes
                    
            else: # for non-modal systems
                
                # Check DOF index is valid
                sys_obj.CheckDOF(DOF)
                
                # Define new constraint equation submatrix
                n = sys_obj.nDOF
                J_new = npy.asmatrix(npy.zeros((n,)))
                J_new[0,DOF] = factor * 1
        
            # Define new constraint equation to link systems 
            sys_obj.AddConstraintEqns(J_new,J_key,checkConstraints=False)
            
        # Use function defined above to process optional arguments
        link_sys(parent_sys,"parent",Xpos_parent,modeshapes_parent,DOF_parent)
        link_sys(child_sys,"child",Xpos_child,modeshapes_child,DOF_child)
            
    
    def freqVals(self,f_salient=None,nf_pad:int=400,fmax=None):
        """"
        Define frequency values to evaluate frequency response G(f) at
        ***
        
        Optional:
        
        * `f_salient`, *array-like* of salient frequencies (Hz)
            
        * `nf`, number of intermediate frequencies between salient points
        """
        
        # Obtain f_salient
        if f_salient is None:
            
            # Peaks are at _damped_ natural frequencies (note: not undamped)
            f_salient = self.CalcEigenproperties()["f_d"]
            f_salient = npy.sort(f_salient)
            
            # Extend beyond min/max f_n value
            f_salient = f_salient.tolist()
            f_salient.insert(0, f_salient[0] - 0.5*(f_salient[1]-f_salient[0]))
            f_salient.append(f_salient[-1] + 0.5*(f_salient[-1]-f_salient[-2]))
        
        # Flatten input
        f_salient = npy.ravel(f_salient)
        
        # Append fmax to list of salient frequencies
        if not fmax is None:
            f_salient = npy.hstack(([-fmax],npy.sort(f_salient),[fmax]))
    
        # Obtain full list of frequencies
        for i in range(len(f_salient)-1):
    
            f1 = f_salient[i]
            f2 = f_salient[i+1]
            df = (f2 - f1)/(nf_pad+1)
            newf = f1 + npy.arange(0,nf_pad+1)*df
            
            if i ==0:
                fVals = newf
                
            else:
                fVals = npy.hstack((fVals,newf))
    
        # Add on end freq
        fVals = npy.hstack((fVals,f2))
                
        return fVals
    
    
    # Define frequency response
    def CalcFreqResponse(self,
                         fVals=None, fmax=None,
                         A=None, B=None, 
                         C=None, D=None,
                         output_names:list=None,
                         verbose=False
                         ):
        """
        Evaluates frequency response G(f) at specified frequencies
        
        Refer 
        [derivation](../references/Frequency response from state-space representation.pdf) 
        for the basis of the implementation.
        
        ***
        
        **Optional:**
            
        * `fVals`: _array-like_ of frequencies (Hz) to evaluate G(f) at.
        If `None` (default) then frequencies list will be obtained using 
        `freqVals()` member function.
        
        * `A`, `B`: allows overriding of system and load matrices 
        held as attributes.
        
        * `C`, `D`: allows custom output matrices to be provided. 
        If None, `output_mtrx` attribute will be used as `C` and `D` 
        will be ignored.
        
        * `output_names`, _list_ of strings defining names of outputs
                
        ***
        
        **Returns:**
        
        * `f_vals`, _array_ of frequency values to which `G_f` relates
        
        * `G_f`, _ndarray_, usually of shape 
          (C.shape[0], B.shape[1], f_vals.shape[0]), 
          i.e. at each frequency there is a matrix described the complex-valued 
          frequency transfer function mapping applied loads to outputs
        
        """
        
        if verbose:
            print("Calculating frequency response matrices..")
        
        # Get key properties of system 
        hasConstraints = self.hasConstraints()
        nDOF_full = self.GetSystemMatrices(unconstrained=False)["nDOF"]
                
        # Handle optional arguments
        if fVals is None:
            fVals = self.freqVals(fmax=fmax)
            
        fVals = npy.ravel(fVals)
        
        # Define state matrix, if not provided via input arg
        if A is None:
            A = self.GetStateMatrix(unconstrained=hasConstraints,
                                    recalculate=True)
            
        # Define load matrix, if not provided via input arg
        if B is None:
            B = self.GetLoadMatrix(unconstrained=hasConstraints,
                                   recalculate=True)
        
        # Define output matrix, if not provided via input arg
        if C is None:
            
            # Get output matrix for full system, if defined
            C, output_names = self.get_output_mtrx(all_systems=True,
                                                   state_variables_only=False)
            
            # Check shape
            expected = 3*nDOF_full
            if C is not None and C.shape[1]!=expected:
                raise ValueError("Error: C matrix of unexpected shape!\n" + 
                                 "C.shape: {0}\n".format(C.shape) + 
                                 "Expected: {0}".format(expected))
                    
        if C is None or C.shape[0]==0:
            
            if verbose:
                print("***\nWarning: no output matrix defined. "+
                      "Output matrix Gf will hence relate to state " + 
                      "displacements and velocities\n***")
            
            C = None
            
        # Define names of outputs 
        if C is None:
                
            # Outputs are (extended) state vector
            output_names =  ["DIS #%d" % i for i in range(nDOF_full)]
            output_names += ["VEL #%d" % i for i in range(nDOF_full)]
            output_names += ["ACC #%d" % i for i in range(nDOF_full)]
            output_names = npy.array(output_names)
                
        # Provide default names to outputs, if not defined above
        if output_names is None:
            output_names =  ["(Unnamed output #%d)" % i 
                             for i in range(C.shape[0])]
                        
        # Define C and D matrices required to compute transfer matrices
        # relating applied loads to state accelerations
        
        # Obtain A and B matrices for the full system
        A_full = self.GetStateMatrix(unconstrained=False,
                                     recalculate=True)
        
        B_full = self.GetLoadMatrix(unconstrained=False,
                                    recalculate=True)
            
        # Define C and D matrices based on rows relating to state accelerations
        C_acc = A_full[nDOF_full:,:] 
        D_acc = B_full[nDOF_full:,:]
    
        # Get nullspace basis matrix (which will already have been calculated)
        if hasConstraints:
            Z = self._Null_J
            zeros_mtrx = npy.zeros_like(Z)
            Z2 = npy.vstack((npy.hstack((Z,zeros_mtrx)),
                             npy.hstack((zeros_mtrx,Z))))
            
        # Loop through frequencies
        Gf_list = []
        
        for i in range(len(fVals)):
            
            # Define jw
            jw = (0+1j)*(2*npy.pi*fVals[i])
            
            # Define G(jw) at this frequency
            if not self.isSparse:
                I = npy.identity(A.shape[0])
                Gf_states = npy.linalg.inv(jw * I - A) @ B
                
            else:
                I = sparse.identity(A.shape[0])
                Gf_states = sparse.linalg.inv(jw * I - A) @ B
                
            # Convert to map loads to state variables of constrained problem
            # i.e. full set of freedoms
            if hasConstraints:
                Gf_states = Z2 @ Gf_states
                
            # Compute matrix to map applied loads to state acceleration
            Gf_acc = C_acc @ Gf_states + D_acc
            
            # Stack to obtain matrix mapping applied loads to states {disp,vel}
            # plus state acceletation
            Gf_states_extended = npy.vstack((Gf_states,Gf_acc))
            
            if C is None:
                
                 Gf_rslt = Gf_states_extended
                
            else:
                
                # Compute matrix to map applied loads to outputs
                Gf_outputs = C @ Gf_states_extended
                
                # Adjust for direct mapping between loads and outputs
                if D is not None:
                    Gf_outputs += D    
            
                Gf_rslt = Gf_outputs
                
            # Store in array
            Gf_list.append(Gf_rslt)
            
        # Convert to numpy ndarray format
        Gf_list = npy.asarray(Gf_list)
                    
        # Return values as class instance
        obj = FreqResponse_Results(f=fVals,
                                   Gf=Gf_list,
                                   output_names=output_names)        
        return obj
    
            
    def PlotSystem(self,ax,v,**kwargs):
        """
        Plot system in deformed configuration
        
        **Required:**
        
        * `ax`, axes object, onto which system will be plotted
        
        * `v`, _array_ of displacement results, defining the position of DOFs
          
        Any additional keyword arguments will be passed to 
        PlotSystem_init_plot() method
        """
        
        self.PlotSystem_init_plot(ax,**kwargs)
        self.PlotSystem_update_plot(v)
        
    
    def PlotSystem_init_plot(self,ax,plot_env=True):
        """
        Method for initialising system displacement plot
        ***
        (Will usually be overriden by derived class methods)
        """
                
        # Variables used to generate plot data
        self.x = npy.arange(self.nDOF)
        self.v_env_max = npy.zeros((self.nDOF,))
        self.v_env_min = npy.zeros_like(self.v_env_max)

        # Define drawing artists
        self.lines = {}
        
        self.lines['v'] = ax.plot([], [],'ro',label='$v(t)$')[0]
    
        self.plot_env = plot_env
        if plot_env:        
            
            self.lines['v_env_max'] = ax.plot(self.x,
                                              self.v_env_max,
                                              color='r',alpha=0.3,
                                              label='$v_{max}$')[0]
            
            self.lines['v_env_min'] = ax.plot(self.x,
                                              self.v_env_min,
                                              color='b',alpha=0.3,
                                              label='$v_{min}$')[0]
        
        # Set up plot parameters
        ax.grid(axis='x')
        ax.axhline(0.0,color='k')
        ax.set_xlim(-0.2, self.nDOF-1+0.2)
        ax.set_xticks(self.x)
        ax.set_xlabel("DOF index")
        ax.set_ylabel("Displacement (m)")
        
    
    def PlotSystem_update_plot(self,v):
        """
        Method for updating system displacement plot given displacements `v`
        ***
        (Will usually be overriden by derived class methods)
        """
        
        # Update envelopes
        self.v_env_max = npy.maximum(v,self.v_env_max)
        self.v_env_min = npy.minimum(v,self.v_env_min)       
        
        # Update plot data
        self.lines['v'].set_data(self.x,v)
        
        if self.plot_env:
            self.lines['v_env_max'].set_data(self.x,self.v_env_max)
            self.lines['v_env_min'].set_data(self.x,self.v_env_min)
        
        return self.lines
        
        
        

# **************** FUNCTIONS *********************
        

def freq_from_angularFreq(omega):
    """
    Returns the frequency (Hz) equivilent to angular frequency `omega` (rad/s)
    
    $$ f = \omega / 2\pi $$
    """
    return omega / (2*npy.pi)


def angularFreq(f):
    """
    Returns the angular frequency (rad/s) equivilent to frequency `f` (Hz)
    
    $$ \omega = 2\pi f $$
    """
    return 2*npy.pi*f

        
def SDOF_stiffness(M,f=None,omega=None):
    """
    Returns the stiffness of SDOF oscillator, given mass and frequency inputs
    
    $$ \omega = 2\pi f $$
    $$ K = \omega^{2}M $$ 
    
    ***
    Required:
    
    * `M`, mass (kg)
    
    ***
    Optional:
        
    * 'f', frequency (Hz)
    
    * `omega`, angular frequency (rad/s)
    
    Either `f` or `omega` must be provided. If both are provided, 
    `f` is used
    """
    
    if f is not None:
        
        if omega is not None:
            if omega != angularFreq(f):
                print("Warning: arguments `f` and `omega` are contradictory")
        
        omega = angularFreq(f)
        
    return M * (omega**2)


def SDOF_dashpot(M,K,eta):
    """
    Returns the dashpot rate of SDOF oscillator given mass, stiffness and 
    damping ratio inputs
    
    $$ \lambda = 2\zeta\sqrt{KM} $$
    
    ***
    Required:
    
    * `M`, mass (kg)
    
    * `K`, stiffness (N/m)
    
    * `eta`, damping ratio (1.0=critical)
    
    """
    
    return (2 * (K*M)**0.5) * eta


def SDOF_dampingRatio(M,K,C):
    """
    Returns the damping ratio (1.0=critical) of SDOF oscillator 
    given mass, stiffness and damping ratio inputs
    
    $$ \zeta = \lambda / 2\sqrt{KM} $$
    
    ***
    Required:
    
    * `M`, mass (kg)
    
    * `K`, stiffness (N/m)
    
    * `C`, dashpot rate (Ns/m)
    
    """
    
    return C  / (2 * (K*M)**0.5 )


def SDOF_frequency(M,K):
    """
    Returns the undamped natural frequency of SDOF oscillator 
    with mass `M` and stiffness `K`
    
    $$ \omega = \sqrt{K/M} $$
    $$ f = \omega / 2\pi $$
    """

    return freq_from_angularFreq((K/M)**0.5)



def null_space(A, rcond=None):
    """
    Copy of source code from 
    https://docs.scipy.org/doc/scipy/
    reference/generated/scipy.linalg.null_space.html
    
    Included in Scipy v1.1.0
    
    For now recreate here
    In future should just use Scipy function!
    """
    
    # Check whether Scipy method can be used
    if parse_version(scipy.__version__) >= parse_version('1.1'):
        
        # Use Scipy method
        Q = scipy.linalg.null_space(A=A,rcond=rcond)
    
    # Otherwise (when v1.0 or less being used) use this method
    else:
        """
        Copy of source code from 
        https://docs.scipy.org/doc/scipy/
        reference/generated/scipy.linalg.null_space.html
        
        Included in Scipy v1.1.0
        
        For now recreate here
        In future should just use Scipy function!
        """
    
        u, s, vh = scipy.linalg.svd(A, full_matrices=True)
        
        M, N = u.shape[0], vh.shape[1]
        
        if rcond is None:
            rcond = npy.finfo(s.dtype).eps * max(M, N)
            
        tol = npy.amax(s) * rcond
        
        num = npy.sum(s > tol, dtype=int)
        
        Q = vh[num:,:].T.conj()
        
    return Q


def calc_state_matrix(M,K,C,Minv=None,isSparse=False):
    """
    Assembles _state matrix_ as used in state-space representation of 
    equation of motion
    
    $$ A = [[0,I],[-M^{-1}K,-M^{-1}C]] $$
        
    where **M** is the system mass matrix, **C** is the system damping 
    matrix, **K** is the system stiffness matrix and **I** is an 
    identity matrix.
    
    ***
    Required:
        
    * `M`, mass matrix **M**, shape [n x n]
    
    * `K`, stiffness matrix **K**, shape [n x n]
    
    * `C`, damping matrix **C**, shape [n x n]
    
    ***
    Optional:
        
    * `Minv`, inverse mass matrix, shape [nxn]; can be supplied to avoid need 
      to calculate inverse of `M` within this function
    
    * `isSparse`, _boolean_, if 'True' sparse matrix methods to be used
    
    ***
    Returns:
        
    * `A`, state matrix, shape [2n x 2n]
    
    * `Minv`, inverse mass matrix, shape [n x n]
    
    """
    
    nDOF = M.shape[0]
    
    if not isSparse:
        
        if Minv is None:
            Minv = npy.linalg.inv(M)
            
        I = npy.identity(nDOF)
        z = npy.zeros_like(I)
        A = npy.bmat([[z,I],[-Minv @ K, -Minv @ C]])
        
    else:
        
        if Minv is None:
            Minv = sparse.linalg.inv(M)
            
        I = sparse.identity(nDOF)
        A = sparse.bmat([[None,I],[-Minv @ K, -Minv @ C]])
                
    return A, Minv


def calc_load_matrix(M,Minv=None,isSparse=False):
    """
    Assembles _load matrix_ as used in state-space representation of 
    equation of motion

    $$ B = [[0],[M^{-1}]] $$
    
    where **M** is the system mass matrix.
    
    ***
    Required:
                
    * `M`, mass matrix, shape [n x n]. Unused if `Minv` provided.
    
    ***
    Optional:
        
    * `Minv`, inverse mass matrix, shape [n x n]; can be supplied to avoid need 
      to calculate inverse of `M` within this function
    
    * `isSparse`, _boolean_, if 'True' sparse matrix methods to be used
    
    ***
    Returns:
        
    * `B`, load matrix, shape [2n x n]
    
    """
    
    if not isSparse:
        if Minv is None:
            Minv = npy.linalg.inv(M)
        B = npy.vstack((npy.zeros_like(Minv),Minv))
        
    else:
        if Minv is None:
            Minv = sparse.linalg.inv(M)
        B = sparse.bmat([[None],[Minv]])
        
    return B, Minv


def transform_to_unconstrained(J,M=None,C=None,K=None):
    """
    Transforms a constrained problem with system matrices (`M`,`C`,`K`) 
    and constraints matrix `J` into a unconstrained problem by projecting 
    system matrices onto the nullspace basis of J
    """
    
    dict_to_return={}
    
    # Solve for null space of J
    Z = null_space(J)
    dict_to_return["Null_J"]=Z

    # Compute modified M, C and K matrices
    if M is not None:
        M = Z.T @ M @ Z
        dict_to_return["M"]=M
        
    if C is not None:
        C = Z.T @ C @ Z
        dict_to_return["C"]=C
        
    if K is not None:
        K = Z.T @ K @ Z
        dict_to_return["K"]=K
    
    return dict_to_return


def solve_eig(M,C,K,J=None,isSparse=False,normalise=True,verbose=True):
    """
    Solves for eigenproperties of _state matrix_ 'A', using scipy.linalg.eig() 
    method
    ***
    
    Where constraints are defined via **J** matrix, system matrices are 
    projected onto the null space of **J**, to give an unconstrained 
    eigenproblem in matrix **A'**, shape [2(n-m) x 2(n-m)], i.e. of reduced 
    dimensions.
    
    Eigenproperties of **A'** are computed and converted to give 
    eigenproperties of **A**.
                         
    ***
    Required:
        
    * `M`, `C`, `K`; system mass, damping and stiffness matrices, 
       all of shape [n x n]
    
    * `J`, rectangular matrix of dimensions [m x n], m<n, defining a set of _m_ 
      independent linear constraints
             
    ***
    Returns:
        
    * `s`, _array_, shape (2n,), eigenvalues of 'A'
    
    * `Y`, _matrix_, shape [2n x 2n], the columns of which are 
      left-eigenvectors of 'A'
    
    * `X`, _matrix_, shape [2n x 2n], the columns of which are 
      right-eigenvectors of 'A'
              
    Note all will in general be complex-valued
      
    """
    
    if J is not None:
        constrained=True
    else:
        constrained=False
        
    if constrained:
        
        # Convert to unconstrained problem 
        mdict = transform_to_unconstrained(J=J,M=M,C=C,K=K)
        Z = mdict["Null_J"]
        M = mdict["M"]
        C = mdict["C"]
        K = mdict["K"]
            
        if verbose: 
            print("Null(J)=Z:\n{0}\n".format(Z))
            print("M':\n{0}\n".format(M))
            print("C':\n{0}\n".format(C))
            print("K:\n{0}\n".format(K))
            
    # Get state matrix to compute eigenproperties of
    A, Minv = calc_state_matrix(M,K,C,isSparse=isSparse)
    if verbose: print("A:\n{0}\n".format(A))
            
    # Solve unconstrained eigenproblem
    s, Y, X = scipy.linalg.eig(a=A,left=True,right=True)
    Y = npy.asmatrix(Y)
    X = npy.asmatrix(X)
    
    # Scipy routine actually returns conjugate of Y
    # Refer discussion here:
    # https://stackoverflow.com/questions/15560905/
    # is-scipy-linalg-eig-giving-the-correct-left-eigenvectors
    Y = Y.conj()
    
    if verbose: print("X:\n{0}\n".format(X))
    if verbose: print("Y:\n{0}\n".format(Y))
            
    # Recover solution in x
    # Recall x = Z.y
    if constrained:
        
        zeros = npy.zeros_like(Z)
        Z2 = npy.vstack((npy.hstack((Z,zeros)),npy.hstack((zeros,Z))))
        if verbose: print("Z2:\n{0}\n".format(Z2))
        X = Z2 @ X
        Y = Z2 @ Y
        
    # Return instance of Eig_Results class to act as container for results
    rslts_obj = Eig_Results(s=s,X=X,Y=Y,normalise=normalise)    
    return rslts_obj





# ********************** TEST ROUTINES ****************************************
# (Only execute when running as a script / top level)
if __name__ == "__main__":
    
    M = npy.array([[20,0,0],[0,40,0],[0,0,400]])
    C = npy.array([[0.1,-0.2,0],[-0.2,0.4,-0.7],[0,-0.7,1.0]])
    K = npy.array([[300,-200,0],[-200,200,0],[0,0,100]])
    sys1 = DynSys(M,C,K,name="sys1")

    M = npy.array([[500]])
    C = npy.array([[0.8]])
    K = npy.array([[600]])
    sys2 = DynSys(M,C,K,name="sys2")
    
    M = npy.array([[20,0],[0,40]])
    C = npy.array([[0.1,-0.2],[-0.2,0.4]])
    K = npy.array([[300,-200],[-200,200]])
    sys3 = DynSys(M,C,K,name="sys3")
    
    sys1.AppendSystem(child_sys=sys2,J_key="sys1-2",DOF_parent=2,DOF_child=0)
    sys1.AppendSystem(child_sys=sys3,J_key="sys1-3",DOF_parent=0,DOF_child=1)
    sys1.PrintSystemMatrices()
    
    d = sys1.GetSystemMatrices(createNewSystem=True)
    
    J = d["J_mtrx"]
    
    full_sys = d["DynSys_full"]
    full_sys.PrintSystemMatrices(printValues=True)
    
    M_constrained = full_sys.GetSystemMatrices(unconstrained=False)["M_mtrx"]
    M_unconstrained = full_sys.GetSystemMatrices(unconstrained=True)["M_mtrx"]
    print("M_constrained:\n{0}".format(M_constrained))
    print("M_unconstrained:\n{0}".format(M_unconstrained))
    
    B_constrained = full_sys.CalcLoadMatrix(unconstrained=False)
    B_unconstrained = full_sys.CalcLoadMatrix(unconstrained=True)
    print("B_constrained:\n{0}".format(B_constrained.shape))
    print("B_unconstrained:\n{0}".format(B_unconstrained.shape))
    
    A_constrained = full_sys.CalcStateMatrix(unconstrained=False)
    A_unconstrained = full_sys.CalcStateMatrix(unconstrained=True)
    print("A_constrained:\n{0}".format(A_constrained.shape))
    print("A_unconstrained:\n{0}".format(A_unconstrained.shape))
    
    
