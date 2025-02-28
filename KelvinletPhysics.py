import vtk
from vtk.util.numpy_support import vtk_to_numpy as vtk2np
import os
from scipy import optimize as opt
import numpy as np
from timeit import default_timer as timer
import pickle as pkl
import torch
from einops import rearrange

class KelvinletPhysics:
    def __init__(self, x0, E=2100, v=0.45, e=0.03, mode=0):
        self.x0 = x0
        self.setProperties(E, v, e)
        self.mode = mode
        self.K_mesh = None
        
    def setProperties(self, E, v, e):
        self.E = E
        self.v = v
        self.e = e # might be more optimal at 0.03
    
    
    @staticmethod
    def Kelvinlet(x,x0,f0=None,E=2100,v=0.45,e=0.05):

        m = np.size(x,1)
        n = np.size(x0,1)
        k = np.size(x,0)
        
        # Compute constants
        a = (1+v)/(2*np.pi*E)
        b = a/(4*(1-v))
        ce = 2*e/(3*a-2*b)
        e2 = e*e
        
        # Initialize Kelvinlet matrix
        K = np.zeros([k*m, k*n])

        xt = torch.tensor(x, dtype=torch.float32)
        x0t = torch.tensor(x0, dtype=torch.float32)

        rt = xt.unsqueeze(2) - x0t.unsqueeze(1)
        ret = torch.sqrt(rt.square().sum(0) + e2)
        w1 = ((a-b)/ret + (a*e2)/(2*ret**3)).expand(k,k,-1,-1)
        w2 = (b/ret**3).expand(k,k,-1,-1)
        Ik = torch.eye(k,dtype=torch.float32).unsqueeze(2).unsqueeze(3).expand(-1,-1,m,n)
        rrt = torch.einsum('aij,bij->abij', rt, rt)

        Kijt =  ce*(w1*Ik + w2*rrt)
        Kt = rearrange(Kijt,'a b i j -> (i a) (j b)')
        K = Kt.numpy()
        
        if f0 is not None:
            u = np.reshape(K @ f0.ravel(),(-1,3))
            SE = (105*np.pi)/(512*ce)*np.sum(np.square(f0))
        else:
            u = None
            SE = None
        
        return K, u, SE
    
    @staticmethod
    def runKelvinlet(x0,xc,uc,E=2100,v=0.45,e=0.05):
        
        start = timer()
        K_mesh = KelvinletPhysics.Kelvinlet(x0.transpose(),xc.transpose(),None,E,v,e)[0];
        end = timer()
        
        # response matrix from active centers to measurement points (which are the locations of the boundary conditions)
        start = timer()
        K_meas = KelvinletPhysics.Kelvinlet(xc.transpose(),xc.transpose(),None,E,v,e)[0];
        end = timer()
        
        # set up the inverse problem
        f0 = np.zeros_like(uc);
        
        def objective(f):
            wreg = 1e-3
            delta = uc - np.reshape(K_meas @ f.ravel(),(-1,3))
            return np.sum(np.square(delta))/np.size(f) + wreg*np.dot(f,f)
        def objGrad(f):
            wreg = 1e-3
            delta = uc - np.reshape(K_meas @ f.ravel(),(-1,3))
            return -2*(delta.ravel() @ K_meas)/np.size(f) + 2*wreg*f
        
        # solve the inverse provlem
        start = timer()
        optstruct = opt.minimize(objective, f0.ravel(), method='CG', jac=objGrad, options={'disp': False})
        end = timer()
        
        f = optstruct.x
        return np.reshape(K_mesh @ f.ravel(),(-1,3));
    
    @staticmethod
    def constructGlobalMatrix(x0, E=2100, v=0.45, e=0.05):
        
        start = timer()
        K_mesh = KelvinletPhysics.Kelvinlet(x0.transpose(),x0.transpose(),None,E,v,e)[0];
        end = timer()
        
        return K_mesh
    
    @staticmethod
    def runKelvinletFromGlobal(K_mesh,globalids,uc):

        m = np.size(globalids,0)
        n = np.size(uc,0)
        k = np.size(uc,1) # spatial degrees of freedom: assume either k == 2 or k == 3
        
        # set up the inverse problem
        f0 = np.zeros_like(uc);
        
        start = timer()
        indexer = np.repeat(globalids.ravel(),k) == True
        K_mesh2 = K_mesh[:,indexer]
        K_meas = K_mesh2[indexer, :]
        end = timer()
        
        def objective(f):
            wreg = 1e-3
            delta = uc - np.reshape(K_meas @ f.ravel(),(-1,3))
            return np.sum(np.square(delta))/np.size(f) + wreg*np.dot(f,f)
        def objGrad(f):
            wreg = 1e-3
            delta = uc - np.reshape(K_meas @ f.ravel(),(-1,3))
            return -2*(delta.ravel() @ K_meas)/np.size(f) + 2*wreg*f
        
        # solve the inverse provlem
        start = timer()
        optstruct = opt.minimize(objective, f0.ravel(), method='CG', jac=objGrad, options={'disp': False})
        end = timer()
        
        f = optstruct.x
        
        # propagate to mesh
        return np.reshape(K_mesh2 @ f.ravel(),(-1,3));
    
    def precomputeResponseMatrix(self):
        self.K_mesh = self.constructGlobalMatrix(self.x0, self.E, self.v, self.e)
    
    def saveResponseMatrix(self,fpath):
        with open(fpath, 'wb') as f:
            pkl.dump(self.K_mesh,f,protocol=4)
    
    def loadResponseMatrix(self,fpath):
        with open(fpath, 'rb') as f:
            self.K_mesh = pkl.load(f)
    
    def getSolution(self,a,b):
        
        if self.mode == 0:
            return self.runKelvinlet(self.x0,a,b,self.E,self.v,self.e)
        else:
            if self.K_mesh is None:
                self.precomputeResponseMatrix()
            return self.runKelvinletFromGlobal(self.K_mesh,a,b)
