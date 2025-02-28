# KelvinletPhysics.py
# Generates regularized Kelvin's solution to deformation of a mesh with displacement boundary conditions
# specified on a small number of interaction vertices (<100 points of interaction).
# This solution is optimized for inference speed when varying the points of interaction over a fixed mesh
# geometry by precomputing a large matrix -- to do so, run with mode==1.
# Inputs (provided to class on initialization and get/set methods):
#   x0:   reference state of mesh at rest (vertex positions: Nx3 ndarray, units in meters)
#   E:    Young's modulus (units in Pa)
#   v:    Poisson's ratio
#   e:    regularization distance (units in m)
#   mode: if mode == 0, computes response matrix on the fly;
#         if mode == 1, precomputes a response matrix for a mesh to accelerate inference speed
#   parallel: if parallel == True, accelerates tensor operations on GPU using Pytorch
#             if parallel == False, computes on CPU instead
#   
# Output (accessed via getSolution() method):
#   uk: predicted deformed mesh state based on Kelvin solution to the observed boundary conditions
# 
parallel = None
try:
    import torch
    from einops import rearrange
    parallel = True
except:
    parallel = False
    print("Could not import pytorch. Unable to support GPU computation.")
import vtk
#from vtk.util.numpy_support import vtk_to_numpy as vtk2np
from vtkmodules.util.numpy_support import vtk_to_numpy as vtk2np
import os
from scipy import optimize as opt
import numpy as np
import math
from timeit import default_timer as timer
import pickle as pkl
import itertools

import networkx as nx # note that to enable Rapids backend for GPU graph acceleration, need to include the %env NX_CUGRAPH_AUTOCONFIG=True line in a Jupyter notebook or set environment variable NX_CUGRAPH_AUTOCONFIG=True in shell before starting Python
import nx_cugraph as nxcg

from matplotlib import pyplot

class KelvinletPhysics:
    def __init__(self, x0, E=2100, v=0.45, e=0.05, mode=0, parallel=parallel, tr=None):
        self.parallel = parallel
        self.setMesh(x0, tr)
        self.setProperties(E, v, e)
        self.mode = mode
        self.K_mesh = None
        self.nxgraph = None
        self.cugraph = None
        self.shortestpaths = None
        self.dmat = None
        
    def setProperties(self, E, v, e):
        self.E = E
        self.v = v
        self.e = e # might be more optimal at 0.03
    
    def setMesh(self, x0, tr):
        self.x0 = x0
        if tr is not None:
            self.tr = tr.astype(int)
            self.updateDimensions()
        else:
            self.tr = tr
    
    def updateDimensions(self):
        start = timer()
        
        # get problem size
        self.n = self.x0.shape[0] # number of vertices
        self.k = self.x0.shape[1] # number of spatial degrees of freedom
        self.m = self.tr.shape[0] # number of tetrahedral
        
        # compute element volumes and Jacobians (constant over mesh geometry)
        if self.parallel:
            print("Processing mesh on {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
            tets_t = torch.tensor(self.tr, dtype=torch.int32)
            x_t = torch.tensor(self.x0, dtype=torch.float32)
            xe_t = x_t[tets_t,:]
            
            S_t = torch.cat((torch.ones( [xe_t.shape[0],xe_t.shape[1],1] ),xe_t),axis=2)
            V_t = torch.linalg.det(S_t)/math.factorial(self.k)
            J_t = torch.linalg.inv(S_t)
            J_t = rearrange(J_t, 'k i j -> i j k')
            
            self.V = V_t.numpy().astype(np.float32)
            self.J = J_t.numpy().astype(np.float32)
        else:
            self.V = np.zeros(self.m) # element volumes
            self.J = np.zeros([self.k+1,self.k+1,self.m]) # element Jacobians
            for elm in range(self.m):
                # loop over the elements
                xe = self.x0[self.tr[elm,:],:]
                
                # get shape function
                S = np.concatenate( ( np.ones([self.k+1,1]), xe ), axis=1 )
                Ve = np.linalg.det(S)/math.factorial(self.k) # volume or area of element in 3D or 2D
                Je = np.linalg.inv(S) # element Jacobian
                
                # store in arrays
                self.V[elm] = Ve
                self.J[:,:,elm] = Je
        
        '''
        # compute the free boundary of the mesh (set of surface triangle vertex ids)
        self.bdry = set()
        for elm in range(self.m): #tet in self.tr:
            tet = sorted(self.tr[elm,:])
            for face in (  (tet[0], tet[1], tet[2]), 
                           (tet[0], tet[2], tet[3]), 
                           (tet[0], tet[1], tet[3]),
                           (tet[1], tet[2], tet[3])  ):
                # if face has already been encountered, then it's not a boundary face
                # hashset makes the check in O(1) (extremely fast)
                if face in self.bdry:
                    self.bdry.remove(face)
                else:
                    self.bdry.add(face) # add the face
        #with open('bdry.txt','w') as f:
        #    np.savetxt(f, np.array(list(self.bdry)).astype(int), newline='\n')
        '''
        
        end = timer()
        print("Mesh precomputation: {} seconds".format(end-start))
        #print("{} vertices, {} elements, {} boundary faces".format(self.n, self.m, len(self.bdry)))
        
    def setGraphDistancesFromElementList(self,tets):
        # tets: a Tx4 matrix of element index lists corresponding to a mesh defined over self.x0
        # creates an adjacency matrix of size NxNx4 with weights corresponding to [x,y,z,d] distances
        # between vertices of self.x0
        
        start = timer()
        
        #print(self.parallel)
        if self.parallel:
            print("Constructing graph on {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
            tets_t = torch.tensor(tets, dtype=torch.int32)
            tets_t = tets_t.sort(1).values
            combs_t = torch.combinations(torch.arange(tets_t.size(1)), r=2)

            edges_t = tets_t[:,:,None].expand(-1,-1,len(combs_t)).transpose(1,2)
            idx = combs_t[None,:,:].expand(len(tets_t), -1, -1)
            edges_t = torch.gather(edges_t, dim=2, index=idx).flatten(0,1)
            edges_t = torch.unique(edges_t,dim=0)
            
            x_t = torch.tensor(self.x0, dtype=torch.float32)
            r_t = torch.abs(x_t[edges_t[:,1],:] - x_t[edges_t[:,0],:])
            d_t = r_t.square().sum(dim=1)
            
            edges = edges_t.numpy().astype(np.int32)
            r = r_t.numpy().astype(np.float32)
            dsqr = d_t.numpy().astype(np.float32)
            self.nxgraph = nx.Graph()
            for i in range(len(edges)):
                self.nxgraph.add_edge(edges[i,0], edges[i,1], weight=dsqr[i], d1=r[i,0], d2=r[i,1], d3=r[i,2])
            
            self.cugraph = nxcg.from_networkx(self.nxgraph, preserve_all_attrs=True)
            # NOTE: see here for function definition: https://github.com/rapidsai/cugraph/blob/branch-24.10/python/nx-cugraph/nx_cugraph/convert.py#L89
        else:
            edges = [list(itertools.combinations(sorted(t), 2)) for t in tets] # each pair of edges in tet
            edges = [item for row in edges for item in row] # flattens to single list
            edges = list(set(edges)) # unique pairs

            #edges = edges[:20] # slice to smaller edge graph for debugging
            
            self.nxgraph = nx.Graph()
            for e in edges:
                r = np.abs([self.x0[e[1],:] - self.x0[e[0],:]]).ravel()
                dsqr = np.sum(r*r)
                self.nxgraph.add_edge(e[0],e[1], weight=dsqr, d1=r[0], d2=r[1], d3=r[2])
        
        end = timer()
        print("Build graph (NetworkX): {} seconds".format(end-start))
        print("{} vertices, {} edges".format(self.nxgraph.number_of_nodes(),self.nxgraph.number_of_edges()))
        
    def clearGraph(self):
        self.nxgraph = None
        self.cugraph = None
        self.shortestpaths = None
        self.dmat = None
    
    def computeInteractionDistances(self,root_idx,leaf_idx):
        # Computes the shortest path distances from points at root_idx to each vertex in graph
        # at leaf_idx nodes
        # Uses a networkx implementation of Dijkstra's algorithm, or uses Floyd-Warshall algorithm if distance matrix is precomputed and stored in self.dmat
        # Inputs:
        # root_idx - a list (Nx1) of graph vertex indices to use as seed points; if None, will compute for all possible seed points
        # leaf_idx - a list (Mx1) of graph vertex indices to use as query points; if None, will assume output for all possible query points
        # Outputs:
        # root_to_leaf - distances from self.x0[root_idx,:] to self.x0[leaf_idx,:] (MxNxk); dim2 contains [d1,d2,...,dk]
        # root_to_all - distances from self.x0[root_idx,:] to self.x0[:,:] (MxNxk); dim2 contains [d1,d2,...,dk]
        if self.nxgraph is None:
            print("NetworkX graph does not exist. Call a setGraphDistances() class function to initialize. Defaulting to Euclidean distances")
            return None, None
        
        print("Computing shortest paths (NetworkX)")
        start = timer()
        
        n = np.size(self.x0,0) # number of seed points
        m = np.size(self.x0,0) # number of query points
        k = np.size(self.x0,1) # number of degrees of freedom
        if root_idx is not None:
            n = len(root_idx)
            root_idx = list(root_idx)
        else:
            root_idx = np.arange(0,n)
        
        
        root_to_all = np.zeros([m,n,k])
        #spmethod = "dijkstra"
        #spmethod = "bellman-ford"
        for i in range(n):
            for dim in range(k):
                #d = nx.shortest_path_length(self.nxgraph, root_idx[i], None, weight="d{}".format(dim+1), method=spmethod)
                d = nx.single_source_dijkstra_path_length(self.nxgraph, root_idx[i], weight="d{}".format(dim+1), backend='networkx')
                #d = nx.single_source_dijkstra_path_length(self.cugraph, root_idx[i], weight="d{}".format(dim+1), backend='cugraph')
                root_to_all[:,i,dim] = [ d[j] for j in range(len(d)) ]
            #    #root_to_all[:,i,dim] = nx.dict_to_numpy_array(nx.shortest_path_length(self.nxgraph, root_idx[i], None, weight="d{}".format(dim+1), method=spmethod))
        
        
        '''
        # determine a cutoff distance beyond which to stop looking for paths
        # for a critical threshold of <5% initial displacement corresponding to c = 0.05, this occurs at re > 0.5*e/c for an incompressible material
        critratio = 0.95 # ignore after displacement falls below this percentage
        critlen = 0.5*self.e/critratio
        root_to_all = critlen * np.ones([m,n,k])
        for i in range(n):
            for dim in range(k):
                d = nx.single_source_dijkstra_path_length(self.nxgraph, root_idx[i], cutoff=critlen, weight="d{}".format(dim+1), backend='networkx')
                #d = nx.single_source_dijkstra_path_length(self.cugraph, root_idx[i], cutoff=critlen, weight="d{}".format(dim+1), backend='cugraph')
                for key, value in d.items():
                    root_to_all[key,i,dim] = value
        '''
        
        if leaf_idx is not None:
            m = len(leaf_idx)
            leaf_idx = list(leaf_idx)
        else:
            leaf_idx = np.arange(0,m)
        root_to_leaf = np.take(root_to_all, leaf_idx, axis=0)
        
        end = timer()
        print("Compute shortest paths (NetworkX): {} seconds".format(end-start))
        
        return root_to_leaf, root_to_all

    def precomputeInteractionDistances(self):
        if self.nxgraph is None:
            print("NetworkX graph does not exist. Call a setGraphDistances() class function to initialize. Defaulting to Euclidean distances")
            return
        
        print("Precomputing all pairs shortest paths (NetworkX)")
        start = timer()
        
        #self.shortestpaths = dict(nx.shortest_path(self.nxgraph, None, None, weight="weight", method="dijkstra"))
        #self.shortestpaths = [ nx.shortest_path(self.nxgraph, i, None, weight="weight", method="dijkstra") for i in range(np.size(self.x0,0)) ]
        #self.shortestpaths = dict(nx.all_pairs_all_shortest_paths(self.nxgraph, weight="weight", method="dijkstra"))
        
        #predecessors, distmat = nx.floyd_warshall_predecessor_and_distance(self.nxgraph, weight="weight")
        
        ndp = dict(nx.all_pairs_dijkstra(self.cugraph, weight="weight", backend="cugraph"))
        
        end = timer()
        print("Completed in: {} seconds".format(end-start))

        #for n, (dist, path) in ndp:
        #    print(ndp[n][dist][path])
        
        print(ndp[0][0][400])
        print(ndp[0][1][400])
        
        return
        
        #print(predecessors)
        #print('\n\n\n')
        self.shortestpaths = {}
        for key in predecessors:
            self.shortestpaths[key] = dict(predecessors[key])
        #print(self.shortestpaths)
        
        #print('\n\n\n')
        
        #print(distmat)
        #print('\n\n\n')
        self.dmat = {}
        for key in distmat:
            self.dmat[key] = dict(distmat[key])
        #print(self.dmat)
        
        
        #self.shortestpaths = np.array([self.shortestpaths[i][j] for i in range(self.nxgraph.number_of_nodes()) for j in range(self.nxgraph.number_of_nodes())],dtype=np.int32)
        #self.dmat = np.array([self.dmat[i][j] for i in range(self.nxgraph.number_of_nodes()) for j in range(self.nxgraph.number_of_nodes())],dtype=np.single)
        
        cast = timer()
        print("Cast NetworkX matrices to dictionary: {} seconds".format(cast-end))
        
        #from pprint import pprint
        #pprint(self.shortestpaths)
        
        #print(len(self.shortestpaths))
        #print(self.shortestpaths[0][1])
        #print(self.shortestpaths)
                
        #n = np.size(self.x0,0) # number of seed points
        #m = np.size(self.x0,0) # number of query points
        #k = np.size(self.x0,1) # number of degrees of freedom
        #self.dmat = np.zeros([m,n,k])
        #for dim in range(k):
        #    self.dmat[:,:,dim] = [ [ nx.path_weight(self.nxgraph, path[i][j], "d{}".format(dim+1)) for i in range(m) ] for j in range(n) ]
        
        #end = timer()
        #print("Precompute all pairs shortest paths (NetworkX): {} seconds".format(end-start))
        
    @staticmethod
    def Kelvinlet(x,x0,f0=None,E=2100,v=0.45,e=0.05,parallel=parallel,graphdistances=None):
        # Gets displacements at query positions from regularized Kelvinlet functions
        # centered at a set of positions x0.
        # Inputs:
        # x - Query positions (kxM)
        # x0 - Kelvinlet centers (kxN)
        # f0 - Kelvinlet forces (kxN)
        # E - Young's modulus (1)
        # v - Poisson's ratio (1)
        # e - Kelvinlet regularization (1)
        # parallel - if True, will attempt to run on GPU. Otherwise, will run on CPU.
        # graphdistances - distances from x to x0 (MxNxk); dim2 contains [dx,dy,dx]
        # Outputs:
        # K - Kelvinlet response matrix K = K(x,x0) such that u(x) = K*f0 (Mk x Nk)
        # u - displacements at each query position x, only computed if f0 is defined (kxM)
        # SE - total strain energy of each Kelvinlet deformation
        #
        # May call this function with f0 = None for precomputation of K(x,x0) without
        # inferring displacements from the force basis
        
        m = np.size(x,1)
        n = np.size(x0,1)
        k = np.size(x,0) # spatial degrees of freedom: assume either k == 2 or k == 3
        
        # Compute constants
        a = (1+v)/(2*np.pi*E)
        b = a/(4*(1-v))
        ce = 2*e/(3*a-2*b)
        e2 = e*e
        
        # Initialize Kelvinlet matrix
        K = np.zeros([k*m, k*n])
        if parallel:
            print("Constructing Kelvinlet on {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
            xt = torch.tensor(x, dtype=torch.float32)
            x0t = torch.tensor(x0, dtype=torch.float32)
            
            rt = None
            if graphdistances is None:
                rt = xt.unsqueeze(2) - x0t.unsqueeze(1) # kxMxN
            else:
                rt = rearrange(torch.tensor(graphdistances, dtype=torch.float32), 'i j k -> k i j')
            ret = torch.sqrt(rt.square().sum(0) + e2) # MxN
            
            w1 = ((a-b)/ret + (a*e2)/(2*ret**3)).expand(k,k,-1,-1)
            w2 = (b/ret**3).expand(k,k,-1,-1)
            Ik = torch.eye(k,dtype=torch.float32).unsqueeze(2).unsqueeze(3).expand(-1,-1,m,n)
            rrt = torch.einsum('aij,bij->abij', rt, rt)
            
            Kijt =  ce*(w1*Ik + w2*rrt)
            Kt = rearrange(Kijt,'a b i j -> (i a) (j b)')
            K = Kt.numpy()
        else:
            if graphdistances is not None:
                for i in range(m):
                    for j in range(n):
                        r = graphdistances[i,j,:] # k x 1
                        re = np.sqrt(np.sum(np.square(r)) + e2) # regularized distance from x to x0 (1)
                        Kij = (a-b)/re * np.eye(k) + b/re**3 * np.outer(r,r) + (a*e2)/(2*re**3) * np.eye(k)
                        K[k*i:k*(i+1),k*j:k*(j+1)] = ce*Kij
            else:
                for i in range(m):
                    for j in range(n):
                        r = x[:,i] - x0[:,j] # k x 1
                        re = np.sqrt(np.sum(np.square(r)) + e2) # regularized distance from x to x0 (1)
                        Kij = (a-b)/re * np.eye(k) + b/re**3 * np.outer(r,r) + (a*e2)/(2*re**3) * np.eye(k)
                        K[k*i:k*(i+1),k*j:k*(j+1)] = ce*Kij
        
        if f0 is not None:
            u = np.reshape(K @ f0.ravel(),(-1,3))
            #SE = (105*np.pi)/(1024*e)*(3*a-2*b)*np.sum(np.square(f0));
            SE = (105*np.pi)/(512*ce)*np.sum(np.square(f0));
            # This energy by multiplying the force density by displacement -- external virtual work...
            # TODO: Try taking gradient of displacement function to get stress and strain
        else:
            u = None
            SE = None
        
        return K, u, SE
    
    @staticmethod
    def runKelvinlet(x0,xc,uc,E=2100,v=0.45,e=0.05,parallel=parallel,gdmesh=None,gdmeas=None):
        # Computes Kelvinlet response for displacement specifications
        # Inputs:
        # x0 - Query positions (e.g. mesh coordinates) of total Kelvinlet response (Mxk)
        # xc - coordinates of active Kelvinlet centers (Nxk)
        # uc - specified displacements of of active Kelvinlet centers (Nxk)
        # E - Young's modulus (1)
        # v - Poisson's ratio (1)
        # e - Kelvinlet regularization (1)
        # parallel - if True, will attempt to run on GPU. Otherwise, will run on CPU.
        # gdmesh - distances from x0 to xc (MxNxk+1)
        # gdmeas - distances from xc to xc (NxNxk+1)
        
        # response matrix from active centers to mesh
        start = timer()
        K_mesh = KelvinletPhysics.Kelvinlet(x0.transpose(),xc.transpose(),None,E,v,e,parallel,gdmesh)[0];
        end = timer()
        print("Construct global matrix: {} seconds".format(end-start))
        
        # response matrix from active centers to measurement points (which are the locations of the boundary conditions) -- for now
        # assume only one interaction location that is not cut through, so ignore graph computation
        start = timer()
        K_meas = KelvinletPhysics.Kelvinlet(xc.transpose(),xc.transpose(),None,E,v,e,parallel,gdmeas)[0];
        end = timer()
        print("Construct local matrix: {} seconds".format(end-start))
        
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
        
        # solve the inverse problem
        start = timer()
        #optstruct = opt.minimize(objective, f0.ravel(), method='BFGS', options={'disp': True})
        #optstruct = opt.minimize(objective, f0.ravel(), method='BFGS', jac=objGrad, options={'disp': True})
        #optstruct = opt.minimize(objective, f0.ravel(), method='CG', options={'disp': True})
        optstruct = opt.minimize(objective, f0.ravel(), method='CG', jac=objGrad, options={'disp': True})
        end = timer()
        print("Optimization: {} seconds".format(end-start))
        
        f = optstruct.x
        
        # propagate to mesh
        return np.reshape(K_mesh @ f.ravel(),(-1,3));
    
    @staticmethod
    def constructGlobalMatrix(x0, E=2100, v=0.45, e=0.05, parallel=parallel, graphdistances=None):
        # Precomputes global Kelvinlet interaction matrix across mesh space, assuming mesh vertices as query positions
        # correspond one-to-one with the potential interation points
        # Inputs:
        # x0 - query and interaction positions (e.g. mesh coordinates) of total Kelvinlet response (Mxk)
        # E - Young's modulus (1)
        # v - Poisson's ratio (1)
        # e - Kelvinlet regularization (1)
        # parallel - if True, will attempt to run on GPU. Otherwise, will run on CPU.
        
        # response matrix from active centers to mesh
        print("Construct global matrix start... please wait...")
        start = timer()
        K_mesh = KelvinletPhysics.Kelvinlet(x0.transpose(),x0.transpose(),None,E,v,e,parallel,graphdistances)[0];
        end = timer()
        print("Construct global matrix: {} seconds".format(end-start))
        
        return K_mesh
    
    @staticmethod
    def runKelvinletFromGlobal(K_mesh,globalids,uc):
        # Computes Kelvinlet response for displacement specifications
        # Inputs:
        # K_mesh - precomputed global Kelvinlet response matrix (kMxkM)
        # globalids - boolean index for which nodes are selected as Kelvinlet centers (Mx1)
        # uc - specified displacements of active Kelvinlet centers (Nxk)
        
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
        print("Sample local matrix: {} seconds".format(end-start))
        
        def objective(f):
            wreg = 1e-3
            delta = uc - np.reshape(K_meas @ f.ravel(),(-1,3))
            return np.sum(np.square(delta))/np.size(f) + wreg*np.dot(f,f)
        def objGrad(f):
            wreg = 1e-3
            delta = uc - np.reshape(K_meas @ f.ravel(),(-1,3))
            return -2*(delta.ravel() @ K_meas)/np.size(f) + 2*wreg*f
        
        # solve the inverse problem
        start = timer()
        #optstruct = opt.minimize(objective, f0.ravel(), method='BFGS', options={'disp': True})
        #optstruct = opt.minimize(objective, f0.ravel(), method='BFGS', jac=objGrad, options={'disp': True})
        optstruct = opt.minimize(objective, f0.ravel(), method='CG', jac=objGrad, options={'disp': True})
        end = timer()
        print("Optimization: {} seconds".format(end-start))
        
        f = optstruct.x
        
        # propagate to mesh
        return np.reshape(K_mesh2 @ f.ravel(),(-1,3));
    
    def precomputeResponseMatrix(self):
        if self.nxgraph is not None:
            graphdistances = self.computeInteractionDistances(None,None)
        self.K_mesh = self.constructGlobalMatrix(self.x0, self.E, self.v, self.e, self.parallel, graphdistances)
    
    def saveResponseMatrix(self,fpath):
        with open(fpath, 'wb') as f:
            pkl.dump(self.K_mesh,f,protocol=4)
    
    def loadResponseMatrix(self,fpath):
        with open(fpath, 'rb') as f:
            self.K_mesh = pkl.load(f)

    def saveGraphDistances(self,fpath1,fpath2,fpath3):
        with open(fpath1, 'wb') as f:
            pkl.dump(self.nxgraph,f,protocol=4)
        with open(fpath2, 'wb') as f:
            pkl.dump(self.shortestpaths,f,protocol=4)
        with open(fpath3, 'wb') as f:
            pkl.dump(self.dmat,f,protocol=4)
            
    def loadGraphDistances(self,fpath1,fpath2,fpath3):
        with open(fpath1, 'rb') as f:
            self.nxgraph = pkl.load(f)
        with open(fpath2, 'rb') as f:
            self.shortestpaths = pkl.load(f)
        with open(fpath3, 'rb') as f:
            self.dmat = pkl.load(f)
            
    def getSolution(self,globalids,uc):
        # Gets regularized Kelvin solution to interaction.
        # Inputs:
        # globalids - boolean vector of indices for which nodes are selected as
        #             interaction points/active boundary conditions (Mx1), with
        #             nnz(globalids) == N
        # uc - the specified displacements of these active interaction points (Nxk)
        
        if self.mode == 0:
            gdmesh = None
            gdmeas = None
            if self.nxgraph is not None:
                nodids = np.flatnonzero(globalids)
                gdmeas, gdmesh = self.computeInteractionDistances(nodids,nodids)
            xc = self.x0[globalids.ravel()==True,:]
            return self.runKelvinlet(self.x0,xc,uc,self.E,self.v,self.e,self.parallel,gdmesh,gdmeas)
        else:
            if self.K_mesh is None:
                self.precomputeResponseMatrix()
            return self.runKelvinletFromGlobal(self.K_mesh,globalids,uc)
        
    def getPrincipalStressStrains(self, u):
        start = timer()
        
        # material properties: compute constants
        mu = self.E/(2*(1+self.v)) # Lame parameter 1
        lam = self.E*self.v/((1+self.v)*(1-2*self.v)) # Lame parameter 2
        # material matrix: [s11;s22;s33;s23;s13;s12] = Ke*[e11;e22;e33;e23;e13;e12]
        Ke = [[lam+2*mu, lam,      lam,      0,  0,  0 ],
                   [lam,      lam+2*mu, lam,      0,  0,  0 ],
                   [lam,      lam,      lam+2*mu, 0,  0,  0 ],
                   [0,        0,        0,        mu, 0,  0 ],
                   [0,        0,        0,        0,  mu, 0 ],
                   [0,        0,        0,        0,  0,  mu]];

        e = None
        s = None
        s1 = None
        s2 = None
        s3 = None
        vms = None

        if self.parallel:
            Ke_t = torch.tensor(Ke, dtype=torch.float32)
            J_t = torch.tensor(self.J, dtype=torch.float32)
            tr_t = torch.tensor(self.tr, dtype=torch.int32)
            u_t = torch.tensor(u, dtype=torch.float32)
            ue_t = u_t[tr_t,:]
            
            du_t = torch.einsum('iba,abj->aij',J_t[1:,:,:],ue_t)
            gstrain_t = 0.5*(du_t + rearrange(du_t, 'a i j -> a j i') + torch.einsum('aij,akl->ail',du_t,du_t))
            gstrainv_t = torch.stack( (gstrain_t[:,0,0],gstrain_t[:,1,1],gstrain_t[:,2,2],gstrain_t[:,0,1]+gstrain_t[:,1,0],gstrain_t[:,1,2]+gstrain_t[:,2,1],gstrain_t[:,2,0]+gstrain_t[:,0,1]), dim=1)
            #gstrainv_t = torch.stack( (gstrain_t[:,0,0],gstrain_t[:,1,1],gstrain_t[:,2,2],gstrain_t[:,0,1],gstrain_t[:,1,2],gstrain_t[:,2,0]), dim=1)
            gstressv_t = torch.einsum('ai,ij->aj',gstrainv_t,Ke_t)
            
            gstress_t1 = torch.stack( (gstressv_t[:,0],gstressv_t[:,3],gstressv_t[:,5]), dim=1)
            gstress_t2 = torch.stack( (gstressv_t[:,3],gstressv_t[:,1],gstressv_t[:,4]), dim=1)
            gstress_t3 = torch.stack( (gstressv_t[:,5],gstressv_t[:,4],gstressv_t[:,2]), dim=1)
            gstress_t = torch.stack( (gstress_t1, gstress_t2, gstress_t3), dim=2)

            lam_t = torch.abs(torch.linalg.eigvals(gstress_t))
            vms_t = torch.sqrt(((lam_t[:,0]-lam_t[:,1])**2 + (lam_t[:,1]-lam_t[:,2])**2 + (lam_t[:,2]-lam_t[:,0])**2)/2)
            
            e = rearrange(gstrain_t,'a i j -> i j a').numpy()
            s = rearrange(gstress_t,'a i j -> i j a').numpy()
            s1 = lam_t[:,0].numpy()
            s2 = lam_t[:,1].numpy()
            s3 = lam_t[:,2].numpy()
            vms = vms_t.numpy()
            
        else:
            e = np.ndarray( (self.k,self.k,self.m) )
            s = np.ndarray( (self.k,self.k,self.m) )
            s1 = np.ndarray(self.m)
            s2 = np.ndarray(self.m)
            s3 = np.ndarray(self.m)
            vms = np.ndarray(self.m)
            for eid in range(self.m):
                # get element info
                Je = self.J[:,:,eid]
                ue = u[self.tr[eid,:],:]
                            
                # compute displacement gradient and Green strain
                du = np.matmul(Je[1:,:],ue)
                gstrain = 0.5*(du + du.transpose() + np.matmul(du,du.transpose()))
                gstrain_v = np.array([gstrain[0,0], gstrain[1,1], gstrain[2,2], gstrain[0,1]+gstrain[1,0], gstrain[1,2]+gstrain[2,1], gstrain[2,0]+gstrain[0,2]])
                gstress_v = np.matmul(Ke, gstrain_v)
                gstress = np.array( [ [gstress_v[0],gstress_v[3],gstress_v[5]],
                                      [gstress_v[3],gstress_v[1],gstress_v[4]],
                                      [gstress_v[5],gstress_v[4],gstress_v[2]] ] )
    
                e[:,:,eid] = gstrain
                s[:,:,eid] = gstress
                v1,lam,v2 = np.linalg.svd(gstress)
                s1[eid] = lam[0]
                s2[eid] = lam[1]
                s3[eid] = lam[2]
            vms = np.sqrt(((s2-s1)**2+(s3-s2)**2+(s1-s3)**2)/2)
        
        end = timer()
        print("Compute element stress and strain: {} seconds".format(end-start))
        
        return e,s,s1,s2,s3,vms

    def getForceResponse(self,ids):
        f = np.zeros(3)
        
        # get the tetrahedra neighboring the active ids
        tetlist = set()
        for idx in ids:
            trs = np.argwhere(np.any(self.tr == idx,axis=1))
            for tr in trs:
                tetlist.add( tuple(tr) )
                
        # compute forces over the surface elements attached to those tetrahedra
        #facect = 0
        for elm in tetlist:
            # find surface triangles of those tetrahedra
            eid = elm[0] # tetlist is a set of singleton tuples of element ids
            tet = sorted(self.tr[eid,:])
            faces_sorted = (  (tet[0], tet[1], tet[2]), 
                              (tet[0], tet[2], tet[3]), 
                              (tet[0], tet[1], tet[3]),
                              (tet[1], tet[2], tet[3])  )
            faces_oriented = np.array( [[self.tr[eid,0], self.tr[eid,1], self.tr[eid,2]], 
                                        [self.tr[eid,0], self.tr[eid,2], self.tr[eid,3]], 
                                        [self.tr[eid,0], self.tr[eid,3], self.tr[eid,1]],
                                        [self.tr[eid,1], self.tr[eid,3], self.tr[eid,2]]] )
            ns = np.zeros([4,3])
            for facenum in range(4):
                face = faces_sorted[facenum]
                # hashset makes the check in O(1) (extremely fast)
                if face in self.bdry:
                    # compute surface normal (scaled by area of triangle, not normalized)
                    xs = self.x0[faces_oriented[facenum,:],:]
                    ns[facenum,:] = 0.5*np.cross(xs[2]-xs[0],xs[1]-xs[0])
                    #facect += 1
                        
            # get element info
            Je = self.J[:,:,eid]
            ue = self.u[self.tr[eid,:],:]
                        
            # compute displacement gradient and Green strain
            du = np.matmul(Je[1:,:],ue)
            gstrain = 0.5*(du + du.transpose() + np.matmul(du,du.transpose()))
            gstrain_v = np.array([gstrain[0,0], gstrain[1,1], gstrain[2,2], gstrain[0,1], gstrain[1,2], gstrain[2,0]])
            gstress_v = np.matmul(self.Ke, gstrain_v)
            gstress = np.array( [ [gstress_v[0],gstress_v[3],gstress_v[5]],
                                  [gstress_v[3],gstress_v[1],gstress_v[4]],
                                  [gstress_v[5],gstress_v[4],gstress_v[2]] ] )
            
            # sum the stress projection onto normal vectors = force contributions over element faces
            f += np.sum(ns @ gstress,axis=0)
            
        #print(facect) # reports the total number of surface faces indicent to idlist... may be useful to check upon integration
        return f
