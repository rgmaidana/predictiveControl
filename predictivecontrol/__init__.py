import numpy as np
from scipy.linalg import block_diag, expm
from cvxopt import matrix as cvxmtx
from cvxopt import solvers

solvers.options['show_progress'] = False

''' Model Predictive Control '''
class MPC:
    def __init__(self, A, B, C, Np=10, Nc=4, umax=1, umin=0, dumax=1, dumin=0, T=0.001, discretize=True, **kwargs):
        #### System State-Space ####
        self.A = A
        self.B = B
        self.C = C
        self.x = np.zeros((self.A.shape[0],2), dtype=np.float)
        self.u = np.zeros((self.B.shape[1],1), dtype=np.float)
        self.y = np.zeros((self.C.shape[0],1), dtype=np.float)
        self.T = T

        #### Discretization ####
        if discretize:
            self.A = expm(self.A*self.T)
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
        self.__r = np.zeros((self.u.shape[0],1))                       # Setpoint (reference, init as 0)
        self.H = self.get_H()

        #### Optimization Restrictions ####
        self.M = self.get_M()
        # Saturation limits must be numpy arrays!
        self.umax, self.dumax = umax, dumax
        self.umin, self.dumin = umin, dumin

        #### Other parameters ####
        self.t = np.array([0,0], dtype=np.float)  # Time vector

    def get_M(self):
        Ma = np.eye(self.__Nc,dtype=np.float)
        Ma = np.r_[Ma, -np.eye(self.__Nc)]
        Ma = np.r_[Ma, np.tril(np.ones((self.__Nc,self.__Nc), dtype=np.float))]
        Ma = np.r_[Ma, -np.tril(np.ones((self.__Nc,self.__Nc), dtype=np.float))]

        Mr = np.empty((Ma.shape[0],0), dtype=np.float)
        Mr = np.c_[Mr, Ma]
        
        for _ in range(1,self.u.shape[0]):
            Mr = np.r_[Mr, Ma]

        M = np.empty((Mr.shape[0],0), dtype=np.float)
        M = np.c_[M, Mr]
        for _ in range(1,self.u.shape[0]):
            M = np.c_[M, Mr]
            
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

    def get_H(self):
        Rb = self.__r[0]*np.eye(self.__Nc)
        for i in range(1,self.u.shape[0]):
            Rb = block_diag(Rb, self.__r[i]*np.eye(self.__Nc))
        H = self.P.T.dot(self.P) + Rb
        return H

    def get_predict_horizon(self):
        return self.__Np
    
    def get_control_horizon(self):
        return self.__Nc

    # To-Do
    # def set_model(self, A, B, C, D=0, dist=0):
    #     ref = self.get_reference()
    #     Np = self.get_predict_horizon()
    #     Nc = self.get_control_horizon()

    #     # Reinitialize
    #     self.__init__(A,B,C,D,dist)
    #     self.set_predict_horizon(Np)
    #     self.set_control_horizon(Nc)
    #     self.set_reference(ref)

    def set_predict_horizon(self, Np):
        self.__Np = Np
        self.F = self.get_F()
        self.P = self.get_P()
        self.H = self.get_H()
        ref = self.__r[:]
        self.__r = np.zeros((self.u.shape[0],1))
        self.set_reference(ref)
        self.x = np.zeros((self.A.shape[0],2), dtype=np.float)
    
    def set_control_horizon(self, Nc):
        self.__Nc = Nc
        self.P = self.get_P()
        self.H = self.get_H()
        self.u = np.zeros((self.u.shape[0],1), dtype=np.float)
        self.M = self.get_M()
    
    def get_reference(self):
        return self.__r[:]

    def set_reference(self, ref):
        # Reference is a list
        for i in range(ref.shape[0]):
            self.__r[i] = ref[i]
        self.H = self.get_H()
        
    def run(self):
        xk = np.r_[self.x[:,-1]-self.x[:,-2], self.y[:,-1]]
        xk = xk.reshape((xk.shape[0],1))

        # Redefine restrictions based on last input
        gamma = self.dumax[0]*np.ones((self.__Nc,1), dtype=np.float)
        gamma = np.r_[gamma, -self.dumin[0]*np.ones((self.__Nc,1), dtype=np.float)]
        gamma = np.r_[gamma, (self.umax[0]-self.u[0,-1])*np.ones((self.__Nc,1))]
        gamma = np.r_[gamma, (-self.umin[0]+self.u[0,-1])*np.ones((self.__Nc,1))]
        for i in range(1,self.u.shape[0]):
            col = self.dumax[0]*np.ones((self.__Nc,1), dtype=np.float)
            col = np.r_[col, -self.dumin[i]*np.ones((self.__Nc,1), dtype=np.float)]
            col = np.r_[col, (self.umax[i]-self.u[i,-1])*np.ones((self.__Nc,1))]
            col = np.r_[col, (-self.umin[i]+self.u[i,-1])*np.ones((self.__Nc,1))]
            gamma = np.r_[gamma, col]

        # Build Rs by stacking m inputs (column vector) Np-times
        Rs = np.tile(self.__r[:], (self.__Np,1))

        # Second term of deltaU equation: -(P'*Rsb*r - P'*F*xk) = -P'*(Rs - F*xk)
        # Multiplied by output weight matrix Q
        f = -self.P.T.dot(Rs-self.F.dot(xk))

        # Set up and solve QP problem
        qp_solver = solvers.qp(cvxmtx(self.H.astype(np.float)), cvxmtx(f.astype(np.float)), cvxmtx(self.M.astype(np.float)), cvxmtx(gamma.astype(np.float)))
        du = np.array(qp_solver['x'])
                    
        # New control output (use only first prediction)
        nc_I = np.eye(self.u.shape[0])
        for _ in range(1,self.__Nc):
            nc_I = np.c_[nc_I, np.zeros((self.u.shape[0], self.u.shape[0]))]
        self.u[:,-1] = nc_I.dot(du)[:,0] + self.u[:,-1]

''' Economic Model Predictive Control
    Inherits from MPC class, with the only modification being the minimization function ''' 
class EMPC(MPC):
    def __init__(self, A, B, C, D, minFun, dist=0, **kwargs):
        MPC.__init__(self, A, B, C, D, dist, **kwargs)
        self.minFun = minFun

    def optimize(self, *args):
        return self.minFun(args)