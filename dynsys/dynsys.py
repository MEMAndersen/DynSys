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

# Other imports


class DynSys:
    """
    Class used to store general properties and methods
    required to characterise a generic dynamic (2nd order) system
    """
    
    description="Generic dynamic system"

    def __init__(self,M,C,K,
                 J=None,
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
            
        * Constraint equations matrix, `J`. Shape must be _[mxn]_
            
        """
        
        # Convert to numpy matrix format
        M = npy.asmatrix(M)
        C = npy.asmatrix(C)
        K = npy.asmatrix(K)
        nDOF = M.shape[0]
        
        if not J is None:
            J = npy.asmatrix(J)
        else:
            J = npy.asmatrix(npy.zeros((0,nDOF)))
                
        if isSparse:
            
            # Convert to scipy.sparse csr matrix format
            M = sparse.csc_matrix(M)
            C = sparse.csc_matrix(C)
            K = sparse.csc_matrix(K)
            J = sparse.csc_matrix(J)
        
        # Store as attributes
        self.dynSys_list=[self]
        """
        List of appended dynSys objects
        ***
        The purpose of this is to allow dynamic systems to be defined as an
        ensemble of `dynSys` class instances (or derived classes)
        
        Note by default `self` is always included in this list
        """
        
        self.M_mtrx = M
        """Mass matrix"""
        
        self.C_mtrx = C
        """Damping matrix"""
        
        self.K_mtrx = K
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
        
        self.J_mtrx = J
        """Constraints matrix"""
        
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
                             M_mtrx=None,
                             C_mtrx=None,
                             K_mtrx=None,
                             J_mtrx=None,
                             nDOF=None,
                             ):
        """
        Function carries out shape checks on supplied system matrices
        
        All should be square and of same dimension
        
        
        Unless optional arguments are specified, checks are carried out on 
        system matrices held as class attributes
        
        """

        # Process optional arguments
        if M_mtrx is None: M_mtrx = self.M_mtrx
        if C_mtrx is None: C_mtrx = self.C_mtrx
        if K_mtrx is None: K_mtrx = self.K_mtrx
        if J_mtrx is None: J_mtrx = self.J_mtrx
        if nDOF is None: nDOF = self.nDOF

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
        
        attr_list = ["M_mtrx", "C_mtrx", "K_mtrx", "J_mtrx",
                     "M_inv", "A_inv",
                     "A_mtrx","B_mtrx"]
        
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
        
    @deprecation.deprecated(deprecated_in="0.1.0",current_version=currentVersion)
    def GetSystemMatrices(self):
        """
        Function is used to retrieve system matrices
        
        Matrices are returned via a dictionary (for flexible usage)
        """
        
        # Create empty dictionay
        d = {}
        
        # Populate dictionary
        d["nDOF"]=self.nDOF
        d["M_mtrx"]=self.M_mtrx
        d["C_mtrx"]=self.C_mtrx
        d["K_mtrx"]=self.K_mtrx
        if hasattr(self,"J_mtrx"):
            d["J_mtrx"]=self.J_mtrx
        
        # Return dictionary
        return d
    
    def AddConstraintEqns(self,Jnew,checkConstraints=True):
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
                
        # Store constraint equation as attribute
        if not self.hasConstraints():
            self.J_mtrx = Jnew
        else:
            if self.isSparse:
                self.J_mtrx = npy.append(self.J_mtrx,Jnew,axis=0)
            else:
                self.J_mtrx = npy.append(self.J_mtrx,Jnew,axis=0)
            
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
        if M is None: M = self.M_mtrx
        if C is None: C = self.C_mtrx
        if K is None: K = self.K_mtrx
        if J is None: J = self.J_mtrx
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
            self.A_mtrx = A

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
            self.B_mtrx = B
            
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
        M = self.M_mtrx
        K = self.K_mtrx
        C = self.C_mtrx
        J = self.J_mtrx
        nDOF = self.nDOF
        isDense = not self.isSparse
        
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
        if not self.isModal or not hasattr(self,"isModal"):
            # Non-modal systems, i.e. DOFs are real-world
            
            if DOF < 0 or DOF >= self.nDOF:
                raise ValueError("Error: requested DOF invalid!")
            
        else:
            # Modal systems
            
            if DOF < 0 or DOF >= self.nDOF_realWorld:
                raise ValueError("Error: requested real-world DOF invalid!")
                
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
    
    def AppendSystem(self,DOF1,DOF2,dynSys2):
        """
        Function is used to join two dynamic systems by constraining
        two dofs to translate together
        ***
        Required arguments:
    
        * `DOF1`, DOF index in parent system 
        * `DOF2`, DOF index in child system
        * `dynSys2`, instance of child system
        
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
        d1 = dynSys1.GetSystemMatrices()
        n1 = d1.get("nDOF")
        M1 = d1.get("M_mtrx")
        C1 = d1.get("C_mtrx")
        K1 = d1.get("K_mtrx")
        J1 = d1.get("J_mtrx")
        if J1 is None:
            m1 = 0                      # no constraints
            J1 = npy.zeros((0,n1))      # null array
        else:
            m1 = J1.shape[0]
        
        d2 = dynSys2.GetSystemMatrices()
        n2 = d2.get("nDOF")
        M2 = d2.get("M_mtrx")
        C2 = d2.get("C_mtrx")
        K2 = d2.get("K_mtrx")
        J2 = d2.get("J_mtrx")
        if J2 is None:
            m2 = 0                      # no constraints
            J2 = npy.zeros((0,n2))      # null array
        else:
            m2 = J2.shape[0]
            
        # ---- Update system matrices to reflect freely appending dynSys2 -----    
        
        # Assemble new mass, damping and stiffness matrices
        z1 = npy.asmatrix(npy.zeros((n2,n1)))
        M_new = npy.bmat([[M1,z1.T],[z1,M2]])
        C_new = npy.bmat([[C1,z1.T],[z1,C2]])
        K_new = npy.bmat([[K1,z1.T],[z1,K2]])
        
        # Assemble new constraints matrix
        J_new1 = npy.hstack((J1,npy.asmatrix(npy.zeros((m1,n2)))))
        J_new2 = npy.hstack((npy.asmatrix(npy.zeros((m2,n1))),J2))
        J_new = npy.vstack((J_new1,J_new2))
        
        # Update calling object attibutes
        dynSys1.dynSysList.append(dynSys2)
        dynSys1.M_mtrx = M_new
        dynSys1.C_mtrx = C_new
        dynSys1.K_mtrx = K_new
        dynSys1.J_mtrx = J_new
        dynSys1.nDOF = n1 + n2
        
        # ---- Define constraint equation to constrain DOF1 and DOF2 -----
        
        # Define dynSys1 part of constraint eqn.
        if dynSys1.isModal:
            Jn1 = dynSys1.modeshapes[DOF1,:]
        else:
            Jn1 = npy.asmatrix(npy.zeros((n1,)))
            Jn1[0,DOF1]=+1
        
        # Define dynSys2 part of constraint eqn.
        if dynSys2.isModal:
            Jn2 = dynSys2.modeshapes[DOF2,:] 
        else:
            Jn2 = npy.asmatrix(npy.zeros((n2,)))
            Jn2[0,DOF2]=-1
            
        # Combine and assign
        Jn = npy.hstack((Jn1,Jn2))
        self.AddConstraintEqns(Jn,checkConstraints=False)  # note by definition new constraint is independent of other pre-defined constraints
    
    
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
    
    pass
        
    