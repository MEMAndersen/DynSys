# -*- coding: utf-8 -*-
"""
Classes used to define linear dynamic systems
"""

from __init__ import __version__ as currentVersion

# Std library imports
import numpy as npy
import pandas as pd
import matplotlib.pyplot as plt
import deprecation # obtain this from pip
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
        
        self.isSparse = isSparse
        """
        Boolean, denotes whether system matrices should be stored and 
        manipulated as sparse matrices
        """
        
        self._J_dict = {}
        """Dict of constraints matrices"""
        
        if J_dict is not None:
            self._J_dict = J_dict
        else:
            self._J_dict[None] = npy.zeros((0,nDOF))
        
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
            
            
    def GetSystemNames(self):
        """
        Returns list of all systems and sub-systems
        """
        return [x.name for x in self.DynSys_list]
        
        
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
                                 + "Shape: {0}".format(J_mtrx))
            
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
        
        attr_list = ["_M_mtrx", "_C_mtrx", "_K_mtrx", "_J_mtrx",
                     "_M_inv", "_A_inv",
                     "_A_mtrx","_B_mtrx"]
        
        print("System matrices for `{0}`:\n".format(self.name))
        
        for attr in attr_list:
        
            if hasattr(self,attr):
                
                val = getattr(self,attr)
                
                print("{0} matrix:".format(attr))
                print(type(val))
                
                if printShapes: print(val.shape)
                if printValues: print(val)
                
                print("")
            
            
    @deprecation.deprecated(deprecated_in="0.1.0",current_version=currentVersion)
    def GetRealWorldDOFs(self,v):
        """
        Returns dofs in real-world coordinate system
        
        (useful for system defined via modal properties)
        """
        
        if self.isModal:
            return self.modeshapes * v
        else:
            return v
        

    def GetSystemMatrices(self,createNewSystem=False):
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
            
            for x in DynSys_list:
                
                if key in list(x._J_dict.keys()):
                    
                    J_mtrx = x._J_dict[key]
                    
                    if not x.isSparse:
                        J_mtrx = sparse.csc_matrix(J_mtrx)
                    
                    J_list.append(J_mtrx)
                    
                else:
                    
                    J_list.append(None)
                
            J_dict[key] = J_list
                
        # Construct matrix from blocks
        J_mtrx_list = list(J_dict.values())
        J_mtrx = bmat(J_mtrx_list).todense()
        
        # Check shapes of new matrices
        self._CheckSystemMatrices(nDOF=nDOF_new,
                                  M_mtrx=M_mtrx,
                                  C_mtrx=C_mtrx,
                                  K_mtrx=K_mtrx,
                                  J_dict={None : J_mtrx})
        
        # Populate dictionary
        d["nDOF"] = nDOF_new
        
        d["M_mtrx"]=M_mtrx
        d["C_mtrx"]=C_mtrx
        d["K_mtrx"]=K_mtrx
        
        d["J_mtrx"]=J_mtrx
        d["J_dict"]=J_dict        
        
        d["isLinear"]=isLinear
        d["isSparse"]=isSparse
        
        # Create new system object, given system matrices
        if createNewSystem:
            DynSys_full = DynSys(M=M_mtrx,
                                 C=C_mtrx,
                                 K=K_mtrx,
                                 J_dict=J_dict,
                                 isLinear=isLinear,
                                 isSparse=isSparse)
            
            d["DynSys_full"]=DynSys_full
        
        # Return dictionary
        return d
    
    def AddConstraintEqns(self,Jnew,key,checkConstraints=True):
        """
        Function is used to append a constraint equation
        ***
        
        Constraint equations must take the following form:
        $$ J\ddot{y} = 0 $$
        
        ***
        Practically this is done by adding a row to the `J_mtrx` of the system
        
        `J_mtrx` has dimensions _[m,n]_ where:
            
        * _m_ denotes the number of constraint equations
        * _n_ denotes the number of DOFS of the system
            
        **Important note**: `J_mtrx` must be *full rank*, i.e. have
        independent constraints.
        
        `CheckConstraints()` can be used to test whether
        constraint equations are independent
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
        self._J_dict[key]=Jnew
            
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
        if M is None: M = self._M_mtrx
        if C is None: C = self._C_mtrx
        if K is None: K = self._K_mtrx
        if J is None: J = self._J_mtrx
        if nDOF is None: nDOF = self.nDOF
    
        # Check shape of system matrices
        self._CheckSystemMatrices(M_mtrx=M,
                                  C_mtrx=C,
                                  K_mtrx=K,
                                  J_mtrx=J,
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
        if M is None: M = self.M_mtrx
        
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
    
    
    def EqnOfMotion(self,x, t, forceFunc):
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
        
        # Obtain inputs from object
        d = self.GetSystemMatrices()
        M = d["M_mtrx"]
        K = d["K_mtrx"]
        C = d["C_mtrx"]
        J = d["J_mtrx"]
        nDOF = d["nDOF"]
        isSparse = d["isSparse"]
        
        isDense = not isSparse
        
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
        
        if self.hasConstraints():
                
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
        if self.hasConstraints():
            
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
        
        if self.J_mtrx is None:
            return False
        else:
            return self.J_mtrx.shape[0]!=0
    
    def CheckConstraints(self,raiseException=True)->bool:
        """
        Check contraint equations are independent
        ***
        Practically this is done by checking that `J_mtrx` is full rank
        """
    
        if not(self.J_mtrx is None):       
            
            J = self.J_mtrx
            m = J.shape[0]
            
            if self.isSparse:
                J = J.todense()
            
            r = npy.linalg.matrix_rank(J)
        
            if m!=r:
                
                errorStr="Error: constraints matrix not full rank!\nJ.shape: {0}\nComputed rank: {1}".format(J.shape,r)
                
                if raiseException:
                    raise ValueError(errorStr)
                else:
                    print(errorStr)
                    
                return False
                    
        return True
    
    def AppendSystem(self,DOF1,DOF2,dynSys2,key):
        """
        Function is used to join two dynamic systems by constraining
        two dofs to translate together
        ***
        Required arguments:
    
        * `DOF1`, DOF index in parent system 
        
        * `DOF2`, DOF index in child system
        
        * `dynSys2`, instance of child system
        
        * `key`, key for new set of constraint equations defined by invoking 
          this function
        
        **Note**: only applies to *linear* systems with constant `M`, `C`, `K`
        system matrices. (This is checked: an exception will be raised if used 
        with nonlinear systems).
        
        """
        
        dynSys1 = self   # for clarity in the following code
        
        # Check selected DOF indices are valid
        dynSys1.CheckDOF(DOF1)
        dynSys2.CheckDOF(DOF2)
        
        # Check systems are linear
        if not dynSys1.isLinear:
            raise ValueError("Error: 'dynSys1' is not linear!")
        if not dynSys2.isLinear:
            raise ValueError("Error: 'dynSys2' is not linear!")
    
        # Retrieve system matrices for systems to be joined
        n1 = dynSys1.nDOF
        n2 = dynSys2.nDOF
        
        # Add system to systems list
        self.DynSys_list.append(dynSys2)
            
        # ---- Define new constraint equation to constrain DOF1 and DOF2 -----
        
        # Define dynSys1 part of constraint eqn.
        attr = "isModal"
        
        if hasattr(dynSys1,attr) and getattr(dynSys1,attr):
            Jn1 = dynSys1.modeshapes[DOF1,:]
        else:
            Jn1 = npy.asmatrix(npy.zeros((n1,)))
            Jn1[0,DOF1]=+1
            
        dynSys1.AddConstraintEqns(Jn1,key,checkConstraints=False)
        
        # Define dynSys2 part of constraint eqn.
        if hasattr(dynSys2,attr) and getattr(dynSys2,attr):
            Jn2 = dynSys2.modeshapes[DOF2,:] 
        else:
            Jn2 = npy.asmatrix(npy.zeros((n2,)))
            Jn2[0,DOF2]=-1
            
        dynSys2.AddConstraintEqns(Jn2,key,checkConstraints=False)
    
    
    def freqVals(self,f_salient=None,nf_pad:int=100,fmax=None):
        """"
        Define frequency values to evaluate frequency response G(f) at
        ***
        
        Optional:
        
        * `f_salient`, *array-like* of salient frequencies (Hz)
            
        * `nf`, number of intermediate frequencies between salient points
        """
        
        # Obtain f_salient
        if f_salient is None:
            f_salient = self.CalcEigenproperties()["f_n"]
            f_salient = npy.sort(f_salient)
            #print("Salient frequencies: \n{0}".format(f_salient))
        
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
                         fVals=None,
                         fmax=None,
                         A_mtrx=None,
                         B_mtrx=None,
                         output_mtrx=None
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
        
        * `A_mtrx`, `B_mtrx`: allows overriding of system and load matrices 
        held as attributes.
        
        * `output_mtrx`: allows custom output matrix (or list of output 
        matrices) to be provided. Otherwise `output_mtrx` attribute will be 
        used, if defined. Otherwise an exception will be raised.
        
        """
        # Handle optional arguments
        if fVals is None:
            fVals = self.freqVals(fmax=fmax)
            
        fVals = npy.ravel(fVals)
        
        if A_mtrx is None:
            A_mtrx = self.CalcStateMatrix()
            
        if B_mtrx is None:
            B_mtrx = self.CalcLoadMatrix()
            
        # Get output matrix
        if output_mtrx is None:
            output_mtrx = self.output_mtrx
                
        # Obtain only the part of the output matrix relating to state variables
        nDOF = self.nDOF
        output_mtrx = output_mtrx[:,:2*nDOF]
        
        if output_mtrx.shape[0]==0:
            print("***\nWarning: no output matrix defined. "+
                  "Identity matrix will be used instead\n"+
                  "Output matrix Gf will hence relate to state variables\n***")
            output_mtrx = npy.identity(2*nDOF)
        
        # Determine number of inputs and frequencies
        Ni = B_mtrx.shape[1]
        nf = len(fVals)
        
        # Determine number of outputs
        No = output_mtrx.shape[0]
        
        # Define array to contain frequency response for each 
        G_f = npy.zeros((No,Ni,nf),dtype=complex)
        
        # Loop through frequencies
        for i in range(len(fVals)):
            
            # Define jw
            jw = (0+1j)*(2*npy.pi*fVals[i])
            
            # Define G(jw) at this frequency
            if not self.isSparse:
                Gf = output_mtrx* npy.linalg.inv(jw * npy.identity(A_mtrx.shape[0]) - A_mtrx) * B_mtrx
            else:
                Gf = output_mtrx * sparse.linalg.inv(jw * sparse.identity(A_mtrx.shape[0]) - A_mtrx) * B_mtrx
                
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
    _ = ax_magnitude
    _.plot(f,npy.abs(G_f),label=label_str) 
    _.set_xlabel("Frequency f (Hz)")
    _.set_ylabel("Magnitude |G(f)|")
    if label_str is not None: _.legend()
    
    # Prepare phase plot
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
    
    sys1.AppendSystem(2,0,sys2,"sys1-2")
    sys1.AppendSystem(0,1,sys3,"sys1-3")
    
    print(sys1.GetSystemNames())
    d = sys1.GetSystemMatrices(createNewSystem=False)
    #full_sys = d["full_DynSys"]
    
    