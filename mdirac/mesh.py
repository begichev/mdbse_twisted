import math
import cmath
import numpy as np
import itertools
import matplotlib.pyplot as plt

class Domain():
    def __init__(self):
        self.Np=None
        self.p=None    
                
    def show(self,inds=[],dpi=80,**kwargs):
        fig,ax=plt.subplots(dpi=dpi)
        ax.scatter(self.p[...,0],self.p[...,1],**kwargs)
        if inds!=[]:
            kwargs.update(c='r')
            ax.scatter(self.p[inds][...,0],self.p[inds][...,1],**kwargs)
        ax.set_aspect(1)

    def savefig(self,path='plots/',inds=[],dpi=80,**kwargs):
        fig,ax=plt.subplots(dpi=dpi)
        ax.scatter(self.p[...,0],self.p[...,1],**kwargs)
        if inds!=[]:
            kwargs.update(c='r')
            ax.scatter(self.p[inds][...,0],self.p[inds][...,1],**kwargs)
        ax.set_aspect(1)
        plt.savefig(path)
        plt.close()

class Subdomain(Domain):
    def __init__(self,inds):
        assert(inds[0].size==inds[1].size)
        self.Np=inds[0].size
        self.inds=inds

class Circle(Subdomain):
    pass

class Parallelogram(Domain):    
    def __init__(self,N1,cell):
        dp0=np.linalg.norm(cell[0])
        dp1=np.linalg.norm(cell[1])
        assert(math.isclose(dp0, dp1, rel_tol=1e-15))
        self.dp=dp0
        self.N1=N1
        self.Np=N1*N1
        self.cell = cell
        self.vcell = np.abs(np.linalg.det(cell))
        inds=N1*np.fft.fftfreq(N1)
        #inds = np.arange(-int(0.5*N1),int(0.5*N1),1)
        ii,jj=np.meshgrid(inds,inds,indexing='ij')
        self.p = np.tensordot(ii,cell[0],axes=0) + np.tensordot(jj,cell[1],axes=0)         
    
    def reciprocal_cell(self):
        icell = 2.*np.pi*np.linalg.inv(self.cell).T
        return Parallelogram(self.N1,icell/self.N1)

    def get_circle(self,fraction=0.5,center=(0,0)):
        assert(fraction<=0.5) # otherwise pair interactions will not fit the parallelogram
        diameter=fraction*0.5*math.sqrt(3.)*self.N1*self.dp
        inds = np.where(np.linalg.norm(self.p-self.p[center], axis=2) < 0.5*diameter)
        circle = Circle(inds)
        circle.p = self.p[inds]
        return circle

    def get_BZ(self,scale):
        # works in dimensionless coordinates in terms of 2pi/a
        # scale = 1 -> full BZ hexagon
        x = self.p[:,:,0]
        y = self.p[:,:,1]
        cond1 = np.abs(x)<scale*0.5
        cond2 = (scale-x)>np.sqrt(3)*np.abs(y)
        cond3 = (scale+x)>np.sqrt(3)*np.abs(y)
        inds = np.where(cond1&cond2&cond3)
        bz = Circle(inds)
        bz.p = self.p[inds]
        return bz

    def get_MBZ(self,scale):
        # works in dimensionless coordinates in terms of 2pi/a
        # scale = k0 -> full MBZ hexagon
        # pi/2 rotated to original BZ
        x = self.p[:,:,0]
        y = self.p[:,:,1]
        cond1 = np.abs(y)<scale*0.5
        cond2 = (scale-y)>np.sqrt(3)*np.abs(x)
        cond3 = (scale+y)>np.sqrt(3)*np.abs(x)
        inds = np.where(cond1&cond2&cond3)
        mbz = Circle(inds)
        mbz.p = self.p[inds]
        return mbz


