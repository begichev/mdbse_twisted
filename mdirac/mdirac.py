import math
import cmath
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

sx=np.array([[0,1],[1,0]])
sy=np.array([[0,1j],[-1j,0]])
sz=np.array([[1,0], [0,-1]])

def Hamiltonian(kx,ky,t):
    """
    input: dimensionless kx,ky in units of 2pi/a in BZ
    output: dimensionless energies and eigenvectors in BZ in units of Eg
    eigenvectors in np.linalg.eigh are written in columns [:,i]
    """
    ham = 0.5*sz+t*kx*sx+t*ky*sy
    return ham
    
def eighMesh(mesh,t):
    E=np.zeros((mesh.Np,2),dtype=np.float64)
    U=np.zeros((mesh.Np,2,2),dtype=np.complex128)
    for i,pi in enumerate(mesh.p):
        ham = Hamiltonian(pi[0],pi[1],t)
        E[i],U[i]=np.linalg.eigh(ham)
    return E,U

def TwistHamiltonian(kx,ky,k0,t,wc,wv):
    # 4 bands twist hamiltonian
    # input kx,ky in 2pi/a, wc,wc,t in Delta
    # output eigenenergies in Delta

    K1 = k0*np.array([-0.5,0])
    K2 = k0*np.array([0.5,0])
    block_mono1 = t*(kx-K1[0])*sx+t*(ky-K1[1])*sy+0.5*sz
    block_mono2 = t*(kx-K2[0])*sx+t*(ky-K2[1])*sy+0.5*sz
    interlayer = np.array([[wc,0],[0,wv]])
    ham = np.block([[block_mono1, np.conj(interlayer)],[interlayer, block_mono2]])
    return ham

def eighTwistMesh(mesh,k0,t,wc,wv):
    E=np.zeros((mesh.Np,4),dtype=np.float64)
    U=np.zeros((mesh.Np,4,4),dtype=np.complex128)
    for i,pi in enumerate(mesh.p):
        ham = TwistHamiltonian(pi[0],pi[1],k0,t,wc,wv)
        E[i],U[i]=np.linalg.eigh(ham)
    return E,U

class ContModel():
    def __init__(self,Vc,psic,wc,Vv,psiv,wv):
        self.list=[Vc,psic,wc,Vv,psiv,wv]
        self.interlayer=np.array([[wc,0],[0,wv]])
        self.intralayer=np.array([[Vc*np.exp(1.j*psic*np.pi/180),0],[0,Vv*np.exp(1.j*psiv*np.pi/180)]])

def TwistDimHamiltonian(kx,ky,dim,k0,t,cm):
    # 4*dim**2 bands twist hamiltonian
    # input kx,ky in 2pi/a, wc,wc,t in Delta
    # output eigenenergies in Delta
    N=dim
    K1 = k0*np.array([-0.5,0])
    K2 = k0*np.array([0.5,0])
    G1M = k0*np.sqrt(3)*np.array([-0.5*np.sqrt(3),0.5]) # g3 zihao, guiqung yu
    G2M = k0*np.sqrt(3)*np.array([0.5*np.sqrt(3),0.5]) #  g1

    dimlist = np.arange(-int(0.5*N),int(0.5*N))
    k1x = np.array([kx+i*G1M[0]+j*G2M[0]-K1[0] for i in dimlist for j in dimlist])
    k1y = np.array([ky+i*G1M[1]+j*G2M[1]-K1[1] for i in dimlist for j in dimlist])
    k2x = np.array([kx+i*G1M[0]+j*G2M[0]-K2[0] for i in dimlist for j in dimlist])
    k2y = np.array([ky+i*G1M[1]+j*G2M[1]-K2[1] for i in dimlist for j in dimlist])
    k1 = np.array([k1x,k1y])
    k2 = np.array([k2x,k2y])
    six = np.array([sx for i in dimlist for j in dimlist])
    siy = np.array([sy for i in dimlist for j in dimlist])
    sii = np.array([six,siy])
    si_k1 = np.einsum('ij,ijnk->jnk',k1,sii)
    si_k2 = np.einsum('ij,ijnk->jnk',k2,sii)
    longsz = np.kron(np.eye(N**2),sz)
    ham_mono1 = t*block_diag(*si_k1.tolist())+0.5*longsz
    ham_mono2 = t*block_diag(*si_k2.tolist())+0.5*longsz

    keys0 = np.identity(N**2)
    keysG2 = np.eye(N**2, k=-1) # 1 step down corresponds to +G2M scattering process (g1)
    keysG1 = np.eye(N**2, k=-N) # N steps down corresponds to G1M scattering process (g3)
    keysG3 = np.eye(N**2, k=-N-1) # correspongs to g1+g3 = -g2, matrix for g2 is then conjugated
    intralayer = cm.intralayer
    intra_scatg1 = np.kron(keysG2, intralayer)
    # minus is from g2 = -(g1+g3)
    intra_scatg2 = np.kron(keysG3, np.conj(intralayer))
    intra_scatg3 = np.kron(keysG1, intralayer)
    interlayer = cm.interlayer
    inter_scat0 = np.kron(keys0, interlayer)
    inter_scatg1 = np.kron(keysG2, interlayer)
    inter_scatg3 = np.kron(keysG1, interlayer)

    intra_scat = intra_scatg1 + intra_scatg2 + intra_scatg3
    inter_scat = inter_scat0 + inter_scatg1 + inter_scatg3

    # l=1 negative sign for the bottom layer
    ham = np.block([[ham_mono1+np.conj(intra_scat), np.conj(inter_scat)],[inter_scat, ham_mono2+intra_scat]])
    return ham

def PlotGKMG(dim,k0,kstep0,t,cm,path='/pics'):
    # plotting levels-levels moire bands GKMG dispersion
    dir_path = path
    N = dim
    stepsGK = int(k0/kstep0)
    stepsKM = int(1/2*k0/kstep0)
    stepsMG = int(np.sqrt(3)/2*k0/kstep0)

    kGKx = 0.5*kstep0
    kGKy = 0.5*np.sqrt(3)*kstep0
    kKMx = -1*kstep0
    kKMy = 0
    kMGx = 0
    kMGy = -1*kstep0
    kx = ky = 0 # start from G
    eigenenergies = []

    AllK = stepsGK+stepsKM+stepsMG
    E = np.zeros((AllK,4*N**2),float)

    for i in np.arange(stepsGK):
        kx += kGKx
        ky += kGKy
        ham = TwistDimHamiltonian(kx,ky,dim,k0,t,cm)
        eigenvalues = np.linalg.eigvalsh(ham)
        E[i] = np.real(eigenvalues)

    for i in np.arange(stepsGK,stepsGK+stepsKM):
        kx += kKMx
        ky += kKMy
        ham = TwistDimHamiltonian(kx,ky,dim,k0,t,cm)
        eigenvalues = np.linalg.eigvalsh(ham)
        E[i] = np.real(eigenvalues)

    for i in np.arange(stepsGK+stepsKM,stepsGK+stepsKM+stepsMG):
        kx += kMGx
        ky += kMGy
        ham = TwistDimHamiltonian(kx,ky,dim,k0,t,cm)
        eigenvalues = np.linalg.eigvalsh(ham)
        E[i] = np.real(eigenvalues)


    fig, ax = plt.subplots(figsize=(9,9))
    for j in range(0,4*N**2):
        plt.plot(np.arange(AllK), E[:,j], linestyle="-", linewidth=2)
    ax.set_ylabel('E, Delta', fontsize=20)
    plt.ylim(-0.75,0.75)
    plt.xticks(np.array([0,stepsGK,stepsGK+stepsKM,stepsGK+stepsKM+stepsMG]),["G","K","M","G"], fontsize=20)
    plt.tight_layout()
    plt.savefig(path+'/GKMG.png',dpi=100)
    plt.close()

def eighTwistDimMesh(mesh,dim,k0,t,cm):
    E=np.zeros((mesh.Np,4*dim**2),dtype=np.float64)
    U=np.zeros((mesh.Np,4*dim**2,4*dim**2),dtype=np.complex128)
    for i,pi in enumerate(mesh.p):
        ham = TwistDimHamiltonian(pi[0],pi[1],dim,k0,t,cm)
        E[i],U[i]=np.linalg.eigh(ham)
    return E,U

def Hbse(E,U,Wkk):
    Nk=E.shape[0]
    H=np.empty((Nk,Nk),dtype=np.complex128)
    for i in range(Nk):
        vi,ci=U[i,0],U[i,1]
        #vi,ci=U[i,:,0],U[i,:,1]
        #somehow commented part gives wrong result, though it looks reliable
        for j in range(Nk):
            vj,cj=U[j,0],U[j,1]
            #vj,cj=U[j,:,0],U[j,:,1]
            vv = np.vdot(vj,vi)
            cc = np.vdot(ci,cj)         
            H[i,j] = -Wkk[i,j]*cc*vv
    for i in range(Nk):
        H[i,i] += E[i,1] - E[i,0]
    return H

def TwistHbse(E,U,Wkk,ind_c,ind_v):
    Nk=E.shape[0]
    H=np.empty((Nk,Nk),dtype=np.complex128)
    for i in range(Nk):
        vi,ci=U[i,ind_c],U[i,ind_v]
        for j in range(Nk):
            vj,cj=U[j,ind_c],U[j,ind_v]
            vv = np.vdot(vj,vi)
            cc = np.vdot(ci,cj)         
            H[i,j] = -Wkk[i,j]*cc*vv
    for i in range(Nk):
        H[i,i] += E[i,ind_c] - E[i,ind_v]
    return H

### <c|operator|v>
def cov_matrix_elements(operator,U,ind_c,ind_v):
    #vk,ck = U[:,ind_v],U[:,ind_c]
    vk,ck = U[:,:,ind_v],U[:,:,ind_c]
    return np.einsum('ki,ij,kj->k',ck.conj(),operator,vk)

def exciton_elements(Ux,cov):
    return np.einsum('nk,k->n',np.transpose(Ux),cov)

def get_sigma_xx(factor,omega,dE,rx):
    """
    rxx is array of computed matrix elements corresponding to transitions from v to c:
    <c|rxx|v> 
    """
    sigma=np.zeros(omega.size,dtype=np.complex128)
    for i in range(dE.shape[0]):
        s=abs(rx[i])**2
        sigma += (s/dE[i])/(omega-dE[i])
    return factor*sigma

def get_sigma_xy(factor,omega,dE,rx, ry):
    sigma=np.zeros(omega.size,dtype=np.complex128)
    for i in range(dE.shape[0]):
        s=rx[i]*np.conj(ry[i])
        sigma += (s/dE[i])/(omega-dE[i])
    return factor*sigma
