import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
        
class Keldysh():
    def __init__(self,epseff,r0,kstep0):
        self.epseff=epseff
        self.r0=r0
        self.kstep0=kstep0
    
    def kpotential(self,absk):
        factor=np.sqrt(3)/(2*self.eps)*(self.kstep0)**2
        return factor/(absk*(1.+self.r0*absk))
        
    # Average value of kpotential in kmesh cell at k=0, ignoring self.r0
    # integral 1/q over hex with side=1 = 3*sqrt(3)*log(3)

    def WK00(self,kmesh):
        #factor=8.*math.pi # circle radius dp/2
        # integral = 3*np.log(3) hexagon 
        integral = 4.254 # parallelogram
        factor = integral/self.eps*self.kstep0
        return factor
    
    
def pairs(subdomain,Wk):
    W=np.empty((subdomain.Np,subdomain.Np),dtype=np.float64)
    for ij0,(i0,j0) in enumerate(zip(*subdomain.inds)):
        for ij1,(i1,j1) in enumerate(zip(*subdomain.inds)):
            W[ij0,ij1]=Wk[i1-i0,j1-j0]
    return W

def sample_to_mesh(interaction,mesh):
    kmesh=mesh
    absk=np.linalg.norm(kmesh.p,axis=2)
    WK=interaction.kpotential(absk)               
    WK[0,0] = interaction.WK00(kmesh)
    return WK