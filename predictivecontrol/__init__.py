import numpy as np
from scipy.linalg import block_diag, expm
from cvxopt import matrix as cvxmtx
from cvxopt import solvers

solvers.options['show_progress'] = False

''' Model Predictive Control '''
class MPC:
    def __init__(self, A, B, C, D=0, dist=0, Np=10, Nc=4, umax=1, umin=0, dumax=1, dumin=0, T=0.001, discretize=True, **kwargs):
        #### System State-Space ####
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dist = dist
        self.T = T

        #### Discretization ####
        if discretize:
            self.A = expm(self.A.dot(self.T))
            self.B = np.linalg.inv(A).dot((self.A - np.eye(self.A.shape[0]))).dot(self.B)
        
        #### Augmented State-Space ####
        self.Aa = np.r_[np.c_[self.A, np.zeros((self.A.shape[0],self.C.shape[0]))], np.c_[self.C.dot(self.A), np.eye(self.C.shape[0])]]
        self.Ba = np.r_[self.B, self.C.dot(self.B)]
        self.Ca = np.c_[self.C, np.eye(self.C.shape[0])]
        
        #### Model Predictive Controller ####
        self.__Np = Np         # Prediction horizon
        self.__Nc = Nc         # Control Horizon
        self.F = self.get_F()
        self.P = self.get_P()
        self.u = np.zeros((self.B.shape[1],self.__Nc), dtype=np.float)
        self.x = np.zeros((self.A.shape[0],self.__Np), dtype=np.float)
        self.__Rs = np.zeros((self.F.shape[0],self.B.shape[1]))                       # Setpoint (reference, init as 0)
        self.H = self.get_H()        
        self.iH = np.linalg.inv(self.H)
        
        #### Optimization Restrictions ####
        self.M = self.get_M()

        # Saturation limits must be numpy arrays!
        self.umax, self.dumax = umax, dumax
        self.umin, self.dumin = umin, dumin

        #### Other parameters ####
        self.t = np.array([0,0], dtype=np.float)  # Time vector

    def get_M(self):
        Ma = np.eye(self.__Nc,dtype=np.float)
        Ma = np.r_[Ma, np.tril(np.ones((self.__Nc,self.__Nc), dtype=np.float))]
        Ma = np.r_[Ma, -np.eye(self.__Nc)]
        Ma = np.r_[Ma, -np.tril(np.ones((self.__Nc,self.__Nc), dtype=np.float))]

        # Ma = np.tril(np.ones((self.__Nc, self.__Nc), dtype=np.float))
        # Ma = np.r_[np.r_[np.r_[np.eye(self.__Nc), Ma], -np.eye(self.__Nc)], -np.eye(Ma.shape[0])*Ma]

        M = np.empty((Ma.shape[0],0), dtype=np.float)
        M = np.c_[M, Ma]
        for _ in range(1,self.u.shape[0]):
            M = np.c_[M, Ma]
        return M

    def get_F(self):
        F = self.Ca.dot(self.Aa)
        for i in range(1,self.__Np):
            F = np.vstack((F, self.Ca.dot(np.linalg.matrix_power(self.Aa,i+1))))
        return F
                
    def get_P(self):
        P = np.zeros((self.Ca.shape[0], self.__Nc*self.Ca.shape[0]))
        P[:,0:self.Ca.shape[0]] = self.Ca.dot(self.Ba)
        for i in range(1,self.__Np):
            row = np.roll(P[-self.Ca.shape[0]:,:],1)
            row[:,0:self.Ca.shape[0]] = (self.Ca.dot(np.linalg.matrix_power(self.Aa,i))).dot(self.Ba)
            P = np.r_[P, row]
        return P

    # Is this correct? For every input, make a diagonal augmented matrix
    def get_H(self):
        H = self.P.T.dot(self.P) + self.__Rs[0,0]*np.eye(self.P.T.dot(self.P).shape[0])
        for i in range(1,self.B.shape[1]):
            mtx = self.P.T.dot(self.P) + self.__Rs[0,i]*np.eye(self.P.T.dot(self.P).shape[0])
            H = block_diag(H, mtx)
        return H

    def get_predict_horizon(self):
        return self.__Np
    
    def get_control_horizon(self):
        return self.__Nc

    def set_model(self, A, B, C, D=0, dist=0):
        ref = self.get_reference()
        Np = self.get_predict_horizon()
        Nc = self.get_control_horizon()

        # Reinitialize
        self.__init__(A,B,C,D,dist)
        self.set_predict_horizon(Np)
        self.set_control_horizon(Nc)
        self.set_reference(ref)

    def set_predict_horizon(self, Np):
        self.__Np = Np
        self.F = self.get_F()
        self.P = self.get_P()
        self.H = self.get_H()
        self.iH = np.linalg.inv(self.H)
        ref = self.__Rs[0,:]
        self.__Rs = ref[0]*np.ones((self.F.shape[0],1))
        for i in range(1,self.B.shape[1]):
            self.__Rs = np.c_[self.__Rs, ref[i]*np.ones((self.F.shape[0]))]
        self.x = np.zeros((self.A.shape[0],self.__Np), dtype=np.float)
    
    def set_control_horizon(self, Nc):
        self.__Nc = Nc
        self.P = self.get_P()
        self.H = self.get_H()
        self.iH = np.linalg.inv(self.H)
        self.u = np.zeros((self.B.shape[1],self.__Nc), dtype=np.float)
        self.M = self.get_M()
    
    def get_reference(self):
        return self.__Rs[0,:]

    def set_reference(self, ref):
        # Reference is a list
        for i in range(self.B.shape[1]):
            self.__Rs[:,i] = (ref[i]*np.ones((self.__Rs.shape[0],1)))[:,0]
        self.H = self.get_H()
        self.iH = np.linalg.inv(self.H)

    def run(self):
        # Redefine restrictions based on last input
        #self.gamma = np.empty((self.M.shape[0],0), dtype=np.float)
        self.gamma = self.dumax[0]*np.ones((self.__Nc,1), dtype=np.float)
        self.gamma = np.r_[self.gamma, (self.umax[0]-self.u[0,-1])*np.ones((self.__Nc,1))]
        self.gamma = np.r_[self.gamma, -self.dumin[0]*np.ones((self.__Nc,1), dtype=np.float)]
        self.gamma = np.r_[self.gamma, (-self.umin[0]+self.u[0,-1])*np.ones((self.__Nc,1))]
        for i in range(1,self.u.shape[0]):
            col = self.dumax[0]*np.ones((self.__Nc,1), dtype=np.float)
            col = np.r_[col, (self.umax[i]-self.u[i,-1])*np.ones((self.__Nc,1))]
            col = np.r_[col, -self.dumin[i]*np.ones((self.__Nc,1), dtype=np.float)]
            col = np.r_[col, (-self.umin[i]+self.u[i,-1])*np.ones((self.__Nc,1))]
            self.gamma = np.c_[self.gamma, col]


        # self.gamma = np.ones((self.B.shape[1],self.__Nc))*self.dumax.reshape((max(self.dumax.shape),1))
        # self.gamma = np.c_[self.gamma, np.ones((self.B.shape[1],self.__Nc))*(self.umax.reshape((max(self.umax.shape),1))-self.u[:,-1].reshape((self.u.shape[0],1)))]
        # self.gamma = np.c_[self.gamma, np.ones((self.B.shape[1],self.__Nc))*(-self.dumin.reshape((max(self.dumax.shape),1)))]
        # self.gamma = np.c_[self.gamma, np.ones((self.B.shape[1],self.__Nc))*(-self.umin.reshape((max(self.umax.shape),1))+self.u[:,-1].reshape((self.u.shape[0],1)))]
        # self.gamma = self.gamma.T

        # Quadratic optimization (returns best solution given restrictions M)
        # New version uses Convex Optimization (cvxopt) python package, as it provides optimized quadratic optimization
        xa = np.array([], dtype=np.float)
        for i in range(self.x.shape[0]):
            xa = np.r_[xa, self.x[i,-1]-self.x[i,-2]]
        for i in range(self.u.shape[0]):
            xa = np.r_[xa, self.x[i,-1]]
        xf = np.array([])
        xf = np.r_[xf, xa]
        for i in range(1,self.u.shape[0]):
            xf = np.c_[xf, xa]        # Add identical columns for other entries?

        xf = xf.reshape((xf.shape[0],self.u.shape[0]))
        f = ((-(self.__Rs-self.F.dot(xf)).T).dot(self.P)).T

        lb, ub = 0, self.H.shape[0]/self.u.shape[0]
        minH = self.H[lb:ub,lb:ub]
        qp_solver = solvers.qp(cvxmtx(minH), cvxmtx(f[:,0]), cvxmtx(self.M), cvxmtx(self.gamma[:,0]))
        du = np.array(qp_solver['x'])
        for i in range(1,self.u.shape[0]):
            lb, ub = i*(self.H.shape[0]/self.u.shape[0]), (i+1)*(self.H.shape[0]/self.u.shape[0])
            minH = self.H[lb:ub,lb:ub]
            qp_solver = solvers.qp(cvxmtx(minH), cvxmtx(f[:,i]), cvxmtx(self.M), cvxmtx(self.gamma[:,i]))
            du = np.c_[du, np.array(qp_solver['x'])]
            
        # New control output (u is bound by control horizon, to avoid memory issues)
        self.u = np.roll(self.u[-self.u.shape[0]:,:],-1)
        self.u[:,-1] = du[0,:] + self.u[:,-2]

''' Economic Model Predictive Control
    Inherits from MPC class, with the only modification being the minimization function ''' 
class EMPC(MPC):
    def __init__(self, A, B, C, D, minFun, dist=0, **kwargs):
        MPC.__init__(self, A, B, C, D, dist, **kwargs)
        self.minFun = minFun

    def optimize(self, *args):
        return self.minFun(args)