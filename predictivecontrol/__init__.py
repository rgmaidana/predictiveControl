import numpy as np
from scipy.linalg import block_diag

''' Model Predictive Control '''
class MPC:
    def __init__(self, A, dA, B, C, D=0, dist=0, Np=10, Nc=4, umax=1, umin=0, dumax=1, dumin=0, T=0.001, dt=1e-3, **kwargs):
        #### Continuous Dynamic System State-Space ####
        self.A = A
        self.dA = dA
        self.B = B
        self.C = C
        self.D = D
        self.dist = dist
        self.T = T

        #### Discrete System State-Space model ####
        A_d = (np.eye(self.A.shape[0]) + (self.A+self.dA).dot(self.T))
        B_d = self.B.dot(self.T)

        #### Discrete Augmented State-Space ####
        self.Aa = np.r_[np.c_[A_d, np.zeros((A_d.shape[0],1))], np.c_[self.C.dot(A_d).reshape((1,A_d.shape[1])), np.ones((1,1))]]
        self.Ba = np.r_[B_d, self.C.dot(B_d).reshape((1,1))]
        self.Ca = np.c_[np.zeros((1,self.Aa.shape[0]-1)), np.array([[1]])]
        
        #### Model Predictive Controller ####
        self.__Np = Np         # Prediction horizon
        self.__Nc = Nc         # Control Horizon
        self.F = self.get_F()
        self.P = self.get_P()
        self.u = np.zeros((self.__Nc), dtype=np.float)
        self.x = np.zeros((self.A.shape[0],self.__Np), dtype=np.float)
        self.__Rs = np.zeros((self.F.shape[0],1))                       # Setpoint (reference, init as 0)
        self.H = self.P.T.dot(self.P) + self.__Rs[0]*np.eye(self.P.T.dot(self.P).shape[0])
        self.iH = np.linalg.inv(self.H)
        
        #### Optimization Restrictions ####
        self.M = np.tril(np.ones((self.__Nc, self.__Nc), dtype=np.float))
        self.M = np.r_[np.r_[np.r_[np.eye(self.__Nc), self.M], -np.eye(self.__Nc)], -np.eye(self.M.shape[0])*self.M]
        self.umax, self.dumax = umax, dumax
        self.umin, self.dumin = umin, dumin
        self.gamma = np.ones((1,self.__Nc*4), dtype=np.float).T

        #### Other parameters ####
        self.t = np.array([0,0], dtype=np.float)  # Time vector

    def get_F(self):
        # Calculate F
        # Calculate the first loop iteration here because we can't concatenate an empty array in python
        aux = self.Ca
        F = self.Aa.dot(np.eye(self.Aa.shape[0]))       
        AA = block_diag(self.Aa, self.Aa)
        for _ in range(1,self.__Np):
            F = AA.dot(np.r_[np.eye(self.Aa.shape[0]), F])
            AA = block_diag(self.Aa, AA)
            aux = block_diag(self.Ca, aux)
        F = aux.dot(F)
        return F
    
    def get_P(self):
        P = np.c_[self.Ca.dot(self.Ba), np.zeros((1,self.__Nc-1), dtype=np.float)]
        for i in range(1,self.__Np):
            row = np.roll(P[-1,:],1).reshape((1,P.shape[1]))
            row[0,0] = self.Ca.dot(np.linalg.matrix_power(self.Aa,i)).dot(self.Ba)
            P = np.r_[P, row]
        return P

    def get_predict_horizon(self):
        return self.__Np
    
    def get_control_horizon(self):
        return self.__Nc

    def set_model(self, A, dA, B, C, D=0, dist=0):
        self.__init__(A,dA,B,C,D,dist)

    def set_predict_horizon(self, Np):
        self.__Np = Np
        self.F = self.get_F()
        self.P = self.get_P()
        self.H = self.P.T.dot(self.P) + self.__Rs[0]*np.eye(self.P.T.dot(self.P).shape[0])
        self.iH = np.linalg.inv(self.H)
        self.__Rs = self.__Rs[0]*np.ones((self.F.shape[0],1))
        self.x = np.zeros((self.A.shape[0],self.__Np), dtype=np.float)
    
    def set_control_horizon(self, Nc):
        self.__Nc = Nc
        self.P = self.get_P()
        self.H = self.P.T.dot(self.P) + self.__Rs[0]*np.eye(self.P.T.dot(self.P).shape[0])
        self.iH = np.linalg.inv(self.H)
        self.M = np.tril(np.ones((self.__Nc, self.__Nc), dtype=np.float))
        self.M = np.r_[np.r_[np.r_[np.eye(self.__Nc), self.M], -np.eye(self.__Nc)], -np.eye(self.M.shape[0])*self.M]
        self.gamma = np.ones((1,self.__Nc*4), dtype=np.float).T
        self.u = np.zeros((self.__Nc), dtype=np.float)

    def get_reference(self):
        return self.__Rs[0]

    def set_reference(self, ref):
        self.__Rs += ref*np.ones(self.__Rs.shape)
        self.H = self.P.T.dot(self.P) + self.__Rs[0]*np.eye(self.P.T.dot(self.P).shape[0])
        self.iH = np.linalg.inv(self.H)

    def optimize(self, iH):
        # QPhild, from Liuping Wang's book
        #
        #  Minimizes the quadratic cost function
        #
        #       J = 0.5 x'Hx + x'f
        #       subject to:  M x < b
        #
        #  where iH = inv(H)
        
        n1 = self.M.shape[0]

        xa = np.array([], dtype=np.float)
        for i in range(self.x.shape[0]):
            xa = np.r_[xa, self.x[i,-1]-self.x[i,-2]]
        xa = np.r_[xa, self.x[0,-1]]
        f = self.F.dot(xa)
        f = f.reshape((f.shape[0],1))
        f = -(self.__Rs-f).T.dot(self.P).T

        # Unconstrained optimal solution is -H/f
        eta = -iH.dot(f)

        # Test if this solution satisfies all restrictions M
        kk = 0
        for i in range(n1):
            if (self.M[i,:].dot(eta) > self.gamma[i] ):
                kk += 1
        if (kk == 0):
            return eta  # If all restrictions are satisfied, we are done!
        
        # If not, we proceed with Hildreth's algorithm
        P = self.M.dot(iH.dot(self.M.T))
        d = (self.M.dot(iH.dot(f)) + self.gamma)
        n = d.shape[0]
        x_ini = np.zeros(d.shape)
        lamb = np.copy(x_ini)
        for _ in range(38):
            lamb_p = np.copy(lamb)
            for i in range(n):
                w = P[i,:].dot(lamb) - P[i,i]*lamb[i,0]
                w += d[i,0]
                la = -w/P[i,i]
                lamb[i,0] = max(0,la)
            
            al = (lamb-lamb_p).T.dot(lamb-lamb_p)
            if (al < 10e-8):
                break
            
        eta -= iH.dot(self.M.T).dot(lamb)
        return eta

    def run(self):
        # Redefine restrictions based on last input
        self.gamma = np.ones((1,self.__Nc))*self.dumax
        self.gamma = np.c_[self.gamma, np.ones((1,self.__Nc))*(self.umax-self.u[-1])]
        self.gamma = np.c_[self.gamma, np.ones((1,self.__Nc))*(-self.dumin)]
        self.gamma = np.c_[self.gamma, np.ones((1,self.__Nc))*(-self.umin+self.u[-1])]
        self.gamma = self.gamma.T

        # Quadratic optimization (returns best solution given restrictions M)
        du = self.optimize(self.iH)

        # New control output (u is bound by control horizon, to avoid memory issues)
        self.u = np.roll(self.u,-1)
        self.u[-1] = du[0] + self.u[-2]

''' Economic Model Predictive Control
    Inherits from MPC class, with the only modification being the minimization function ''' 
class EMPC(MPC):
    def __init__(self, A, B, C, D, minFun, dist=0, **kwargs):
        MPC.__init__(self, A, B, C, D, dist, **kwargs)
        self.minFun = minFun

    def optimize(self, *args):
        return self.minFun(args)