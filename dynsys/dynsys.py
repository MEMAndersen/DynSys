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

import deprecation # not in anaconda distribution - obtain this from pip
#@deprecation.deprecated(deprecated_in="0.1.0",current_version=currentVersion)

import scipy.sparse as sparse

from scipy.linalg import block_diag
from scipy.sparse import bmat

# Other imports


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
                 showMsgs=True):
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
            output_mtrx = npy.asmatrix(npy.zeros((0,3*self.nDOF)))
            
        self.output_mtrx = output_mtrx
        """
        Output matrix
        
        Use `AddOutputMtrx()` to define append output matrices. 
        `CheckOutputMtrx()` can be used to check the validity (shape) of the 
        output matrices defined.
        """
        
        if output_names is None:
            output_names = []
        
        self.output_names = output_names
        """
        List of string descriptions for rows of `output_mtrx`.
        Used to label plots etc.
        """
        
        # Check definitions are consistent
        self._CheckSystemMatrices()
        self.CheckOutputMtrx()
        
        if showMsgs:
            print("%s `%s` initialised." % (self.description,self.name))
        
        if isSparse:
            print("Note: sparse matrix functionality as provided by Scipy "
                  "will be used for system matrices")
            
        
    def _CheckSystemMatrices(self,
                             nDOF=None,
                             M_mtrx=None,
                             C_mtrx=None,
                             K_mtrx=None,
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
        for key, J_mtrx in J_dict.items():
            
            if J_mtrx.shape[1]!=nDOF:    
                raise ValueError("Error: J matrix column dimension inconsistent!\n"
                                 + "Shape: {0}\n".format(J_mtrx.shape)  
                                 + "J_mtrx: {0}".format(J_mtrx))
            
        return True
    
    def CheckOutputMtrx(self,
                        output_mtrx=None,
                        output_names=None):
        """
        Checks that all defined output matrices are of the correct shape
        """
        
        # Handle option arguments
        if output_mtrx is None:
            output_mtrx = self.output_mtrx
            
        if output_names is None:
            output_names = self.output_names
                
        # Check list lengths agree
        
        if len(output_names)!=output_mtrx.shape[0]:
            raise ValueError("Length of lists `output_names` "+
                             "and rows of `output_mtrx` do not agree!\n"+
                             "len(output_names)={0}\n".format(len(output_names))+
                             "output_mtrx.shape: {0}".format(output_mtrx.shape))
        
        # Check shape of output matrix
        if output_mtrx is not None:
            
            nDOF_expected = 3*self.nDOF
            if output_mtrx.shape[1] != nDOF_expected:
                raise ValueError("output_mtrx of invalid shape defined!\n" +
                                 "Shape provided: {0}\n".format(output_mtrx.shape) +
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
    
    
    def AddOutputMtrx(self,
                      output_mtrx=None,
                      output_names=None,
                      fName='outputs.csv'):
        """
        Appends `output_mtrx` to `outputsList`
        ***
        
        `output_mtrx` can either be supplied directly or else read from file 
        (as denoted by `fName`)
        
        `output_mtrx` can be a list of matrices; each will be appended in turn
        
        """
        
        # Read from file if no output_mtrx provided
        if output_mtrx is None:
            output_mtrx, output_names = self.ReadOutputMtrxFromFile(fName)
        else:
            output_mtrx = npy.asmatrix(output_mtrx)
            
        # Create default output names if none provided
        if output_names is None:
            output_names = ["Response {0}".format(x) for x in range(output_mtrx.shape[0])]
            
        # Append to attribute
        self.output_mtrx = npy.append(self.output_mtrx,output_mtrx,axis=0)
        self.output_names += output_names
        
        # Check dimensions of all output matrices defined
        self.CheckOutputMtrx()
    
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
    

    def GetSystemMatrices(self,createNewSystem=True):
        """
        Function is used to retrieve system matrices, which are not usually to 
        be accessed directly, except by member functions
        
        ***
        A dict is used to return matrices
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
            
        # Assemble full matrix
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
        
        # Populate dictionary
        d["nDOF"] = nDOF_new
        
        d["M_mtrx"]=M_mtrx
        d["C_mtrx"]=C_mtrx
        d["K_mtrx"]=K_mtrx
        
        d["J_dict"]=J_dict        
        d["J_mtrx"]=J_mtrx
        
        d["isLinear"]=isLinear
        d["isSparse"]=isSparse
        
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
                        applyConstraints:bool=False,
                        saveAsAttr:bool=True,
                        M=None,
                        C=None,
                        K=None,
                        J=None,
                        nDOF=None):
        """
        Assembles the continous-time state matrix `A_mtrx` used in state-space 
        methods
        ***
        
        The continuous-time state matrix is as follows:
            
        $$ A = [[0,I],[-M^{-1}K,-M^{-1}C]] $$
        
        where **M** is the system mass matrix, **C** is the system damping 
        matrix, **K** is the system stiffness matrix and **I** is an 
        identity matrix.
        
        When constraint equations **J** are defined, the following augmented 
        form is obtained:
            
        $$ A' = [[A, J^{T}],[J,0]] $$
        
        Given system matrices are of shape _[N,N]_, **A** is of shape _[2N,2N]_. 
        If _m_ constraints are defined then **A'** is of shape _[2N+m,2N+m]_.
        
        ***
        Optional:
            
        _Unless optional arguments are specified, system matrices stored as 
        class attributes will be used._
            
        * `M`, mass matrix
        
        * `C`, damping matrix
        
        * `K`, stiffness matrix
        
        * `J`, constraints matrix
        
        * `applyConstraints`: if `True` then augmentated state matrix 
            A = [[A,J.T],[J,0]] will be returned
            
        * `saveAsAttr`: if `True` state matrix returned will also be saved as 
          an object instance attribute
        
        """
        
        # Retrieve system matrices
        d = self.GetSystemMatrices()
        
        # Handle optional arguments
        if M is None: M = d["M_mtrx"]
        if C is None: C = d["C_mtrx"]
        if K is None: K = d["K_mtrx"]
        if J is None: J = d["J_mtrx"]
        if nDOF is None: nDOF = d["nDOF"]
    
        # Check shape of system matrices
        self._CheckSystemMatrices(M_mtrx=M,
                                  C_mtrx=C,
                                  K_mtrx=K,
                                  nDOF=nDOF)
        
        # Assemble state matrix A=[[0,I],[-Minv*K,-Minv*C]]
        
        if not self.isSparse:
            Minv = npy.linalg.inv(M)
            I = npy.identity(nDOF)
        else:
            Minv = sparse.linalg.inv(M)
            I = sparse.identity(nDOF)
        
        if not self.isSparse:
            _A = [[npy.zeros_like(I),I],[-Minv @ K, -Minv @ C]]
            A = npy.bmat(_A)
            
        else:
            _A = [[None,I],[-Minv @ K, -Minv @ C]]
            A = sparse.bmat(_A)
        
        if applyConstraints:
            
            # Augment with constraints
            _A = [[A,J.T],[J,None]]
        
            if not self.isSparse:
                npy.bmat(_A)
                
            else:
                A = sparse.bmat(_A)

        # Save as attribute
        if saveAsAttr:
            self._A_mtrx = A

        return A     
    
    def GetStateMatrix(self,forceRecalculate=False):
        """
        Helper function to obtain state matrix, if already calculated 
        and held as attribute. Otherwise state matrix will be recalculated
        """
        if forceRecalculate:
            return self.CalcStateMatrix()
        
        else:
            attr = "_A_mtrx"
            if hasattr(self,attr):
                return getattr(self,attr)
            else:
                return self.CalcStateMatrix()
    
    def CalcLoadMatrix(self,
                       saveAsAttr=True,
                       M=None):
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
        
        * `saveAsAttr`: if `True` state matrix returned will also be saved as 
          an object instance attribute
        
        """
        
        # Retrieve system matrices
        if M is None: M = self._M_mtrx
        
        self._CheckSystemMatrices(M_mtrx=M)
        
        # Assemble load matrix B=[[0],[M]]
        if not self.isSparse:
            Minv = npy.linalg.inv(M)
        else:
            Minv = sparse.linalg.inv(M)
            
        
        
        if not self.isSparse:
            B1 = npy.zeros_like(Minv)
            B2 = Minv
            B = npy.vstack((B1,B2))
            
        else:
            _B = [[None],[Minv]]
            B = sparse.bmat(_B)
        
        # Save as attribute
        if saveAsAttr:
            self._B_mtrx = B
            
        return B
    
    
    def GetLoadMatrix(self,forceRecalculate=False):
        """
        Helper function to obtain load matrix, if already calculated 
        and held as attribute. Otherwise load matrix will be recalculated
        """
        if forceRecalculate:
            return self.CalcLoadMatrix()
        
        else:
            attr = "_B_mtrx"
            if hasattr(self,attr):
                return getattr(self,attr)
            else:
                return self.CalcLoadMatrix()
            
    
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
        
        # Calculate net force (excluding constraint forces)
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
    
        
    def CalcEigenproperties(self,
                            normaliseEigenvectors=True,
                            ax=None,
                            showPlots=False):
        """
        General method for determining damped eigenvectors and eigenvalues 
        of system
        ***
        
        Note in general eigenproperties will be complex due to non-proportional
        damping.
        
        Eigendecomposition of the system state matrix `A_mtrx` is carried out to 
        obtain eigenvalues and displacement-velocity eigenvectors.
        
        
        **Important Note**: this function cannot (currently) be used for 
        systems with constraint equations. An exception will be raised.
        
        """
        
        # Check for constraints
        if self.hasConstraints():
            raise ValueError("Error: cannot (currently) use function " + 
                             "'CalcEigenproperties' for systems " + 
                             "with constraints"
                             )
        
        # Assemble continous-time state matrix A
        A_c = self.CalcStateMatrix()
        
        if self.isSparse:
            A_c = A_c.todense()
        
        # Eigenvalue decomposition of continuous-time system matrix A
        s,X = npy.linalg.eig(A_c)               # s is vector of singular values, columns of X are right eigenvectors of A
        s_left,Y = npy.linalg.eig(A_c.T)        # s is vector of singular values, columns of Y are left eigenvectors of A
        X = npy.asmatrix(X)
        Y = npy.asmatrix(Y)
        
        # Determine left and right eigenvalues which correspond to one-another
        right_indexs = npy.argsort(s)
        left_indexs = npy.argsort(s_left)
        s = s[right_indexs]
        s_left = s_left[left_indexs]
        X = X[:,right_indexs]
        Y = Y[:,left_indexs]
                
        if normaliseEigenvectors:
            X,Y = self._NormaliseEigenvectors(X,Y)
        
        # Extract modal properties of system from eigenvalues
        f_n_abs, f_n, f_d, eta = self._RealEigenvalueProperties(s)

        # Sort eigenvalues into ascending order of f_n_abs
        i1 = npy.argsort(f_n_abs)
        s = s[i1]
        X = X[:,i1]
        Y = Y[:,i1]
        f_n=f_n[i1]
        eta=eta[i1]
        f_d=f_d[i1]
        
        # Write results to object
        self.s = s
        self.X = X
        self.Y = Y
        self.f_n = f_n
        self.f_d = f_d
        self.eta = eta
        
        if showPlots:
            self._OrthogonalityPlot(ax=ax)
            self._EigenvaluePlot(ax=ax,plotType=1)
            self._EigenvaluePlot(ax=ax,plotType=2)
            self._EigenvaluePlot(ax=ax,plotType=4)
        
        # Return complex eigensolution as dict
        d={}
        d["s"]=s
        d["X"]=X
        d["Y"]=Y
        d["f_n"]=f_n
        d["w_n"]=angularFreq(f_n)
        d["eta"]=eta
        d["f_d"]=f_d
        d["w_d"]=angularFreq(f_d)
        
        return d 
    
    def _RealEigenvalueProperties(self,s=None):
        """
        Recovers real-valued properties from complex eigenvalues
        """
        
        if s is None:
            s = self.s
        
        f_d = npy.imag(s) / (2*npy.pi)             # damped natural frequency
        eta = - npy.real(s) / npy.absolute(s)      # damping ratio
        
        f_n_abs = npy.absolute(s) / (2*npy.pi)     # undamped natural frequency
        f_n = npy.sign(f_d) * f_n_abs              # recovers sign of frequency
        
        return f_n_abs, f_n, f_d, eta
     
    def _NormaliseEigenvectors(self,X=None,Y=None):
        """
        Normalise eigenvectors such that YT.X=I
        
        Optional arguments:
        
        * `X`: Numpy matrix, the columns of which are right-eigenvectors
        * `Y`: Numpy matrix, the columns of which are left-eigenvectors
        
        """
        
        if X is None:
            X = self.X
            
        if Y is None:
            Y = self.Y
        
        d = npy.diagonal(Y.T * X)**0.5
        
        if d.any() != 0:
            X = X / d
            Y = Y / d
        
        self.X = X
        self.Y = Y
                    
        return X, Y
    
    def _EigenvaluePlot(self,
                        ax=None,
                        plotType=1,
                        s=None):
        """
        Plots eigenvalues (assumed to be complex) on the complex plane
        ***
        
        Allowable values for `plotType`:
            
        * `1`: Eigenvalues plotted on complex plane
        
        * `2`: |f_n| values plotted against index
        
        * `4`: Undamped natural frequencies vs damping ratio
        
        """
        
        if s is None:
            s = self.s
            
        if ax is None:
            # Produce new plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.gcf()
            
        f_n_abs,f_n,f_d,eta = self._RealEigenvalueProperties(s)
            
        if plotType == 1:
            
            ax.plot(npy.real(s),npy.imag(s),'.b')
            #ax.axis('equal')
            ax.set_title("Eigenvalues plotted on complex plane")
            ax.set_xlabel("Real component")
            ax.set_ylabel("Imag component")
            
        elif plotType == 2:
            
            ax.plot(range(len(f_n_abs)),f_n_abs)
            ax.set_xlabel("Mode index")
            ax.set_ylabel("Undamped natural frequency (Hz)")
            
        elif plotType == 4:
            
            ax.plot(f_n,eta,'.b')
            
        else:
            raise ValueError("Error: unexpected plotType requested!")
            
    
    def _OrthogonalityPlot(self,ax=None,X=None,Y=None):
        """
        Pixel plot of Y.T*X
        ***
        
        This can be used to check orthogonality of X and Y column vectors
        
        """
        
        if X is None:
            X = self.X
            
        if Y is None:
            Y = self.Y
            
        if ax is None:
            # Produce new plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.gcf()
            
        # Produce pixel plot
        im = ax.imshow(npy.absolute(Y.T*X),interpolation='none',cmap='Greys')
        fig.colorbar(im)
        ax.set_title("Y.T * X product")
    
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
                         force_accn:bool=False,
                         verbose=True
                         ):
        """
        Evaluates frequency response G(f) at specified frequencies
        
        Refer 
        [derivation](../references/Frequency response from state-space representation.pdf) 
        for the basis of the implementation.
        
        ***
        
        Optional:
            
        * `fVals`: _array-like_ of frequencies (Hz) to evaluate G(f) at.
        If `None` (default) then frequencies list will be obtained using 
        `freqVals()` member function.
        
        * `A`, `B`: allows overriding of system and load matrices 
        held as attributes.
        
        * `C`, `D`: allows custom output matrices to be provided. 
        If None, `output_mtrx` attribute will be used as `C` and `D` 
        will be ignored.
        
        * `force_accn`, _boolean_,  can be set to override the above behaviour 
        and obtain frequency transfer matrix relating applied forces to 
        accelerations of the DOFs. E.g. for modal systems, modal acceleration 
        transfer matrix can be obtained in this way
        
        """
        nDOF = self.nDOF
        
        # Handle optional arguments
        if fVals is None:
            fVals = self.freqVals(fmax=fmax)
            
        fVals = npy.ravel(fVals)
        
        if A is None:
            A = self.GetStateMatrix()
            
        if B is None:
            B = self.GetLoadMatrix()
            
        # Get output matrices
        if C is None:
            
            C = self.output_mtrx
                    
        if C.shape[0]==0:
            if verbose:
                print("***\nWarning: no output matrix defined. "+
                      "Output matrix Gf will hence relate to state " + 
                      "displacements and velocities\n***")
            
            C = npy.identity(2*nDOF)
            
        # Retain only columns relating to state variables (disp, vel)
        C = C[:,:2*nDOF]
            
        # Override the above 
        if force_accn:
            
            #print("`force_accn` invoked; frequency response function relates" + 
            #      " applied forces to DOF accelerations")
            
            # Get system matrices and output matrices
            nDOF = self.nDOF
            C = A[nDOF:2*nDOF,:] # rows relating to accelerations
            D = B[nDOF:2*nDOF,:] # rows relating to accelerations
        
        # Determine number of inputs and frequencies
        Ni = B.shape[1]
        nf = len(fVals)
        
        # Determine number of outputs
        No = C.shape[0]
        
        # Define array to contain frequency response for each 
        G_f = npy.zeros((No,Ni,nf),dtype=complex)
        
        # Loop through frequencies
        for i in range(len(fVals)):
            
            # Define jw
            jw = (0+1j)*(2*npy.pi*fVals[i])
            
            # Define G(jw) at this frequency
            if not self.isSparse:
                I = npy.identity(A.shape[0])
                Gf = C * npy.linalg.inv(jw * I - A) * B
                
            else:
                I = sparse.identity(A.shape[0])
                Gf = C * sparse.linalg.inv(jw * I - A) * B
                
            if D is not None:
                Gf += D    
            
            # Store in array
            G_f[:,:,i] = Gf
        
        # Return values
        return fVals, G_f
    
    def PlotSystems_all(self,ax,**kwargs):
        """
        Generic plotter function to display current configuration of 
        all systems and subsystems
        """
        return None
    
    def PlotSystem(self,ax,**kwargs):
        """
        Generic plotter function to display current configuration of this 
        `dynSys` object
        """
        return None


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




def PlotFrequencyResponse(f,G_f,
                          positive_f_only:bool=True,
                          label_str:str=None,
                          plotMagnitude:bool=True,ax_magnitude=None,
                          plotPhase:bool=True,ax_phase=None,
                          f_d:list=None
                          ) -> dict:
    """
    Function to plot frequency response (f,G_f)
    
    ***
    Required:
        
    * `f`, _array-like_, frequency values to which `G_f` relates
    
    * `G_f`, _array_like_, frequency response values corresponding to `f`
    
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
        _ = ax_magnitude
        _.plot(f,npy.abs(G_f),label=label_str) 
        _.set_xlim([fmin,fmax])
        _.set_xlabel("Frequency f (Hz)")
        _.set_ylabel("Magnitude |G(f)|")
        if label_str is not None: _.legend()
    
    # Prepare phase plot
    if plotPhase:
        _ = ax_phase
        _.plot(f,npy.angle(G_f),label=label_str)
        _.set_xlim([fmin,fmax])
        _.set_ylim([-npy.pi,+npy.pi]) # angles will always be in this range
        _.set_xlabel("Frequency f (Hz)")
        _.set_ylabel("Phase G(f) (rad)")
        if label_str is not None: _.legend()
    
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
    
    d = sys1.GetSystemMatrices()
    
    full_sys = d["DynSys_full"]
    full_sys.PrintSystemMatrices(printValues=True)
    d = full_sys.GetSystemMatrices()
