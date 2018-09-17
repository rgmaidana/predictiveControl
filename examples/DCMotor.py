import numpy as np

class Motor:
    def __init__(self, Ra=8, La=170e-3, J=10e-3, b=3e-3, If=0.5, kt=0.521, kw=0.521, T=0.001, **kwargs):
        # Constructive parameters
        self.Ra = Ra
        self.La = La
        self.J = J
        self.b = b
        self.If = If
        self.kt = kt
        self.kw = kw

        # Motor continuous-time state-space
        self.A = np.array([[-self.b/self.J,      self.kt*self.If/self.J],
                           [-self.kw*self.If/self.La, -self.Ra/self.La]])
        self.dA = np.array([[0, 1], [1, 0]]).dot(0.5)  # Derivative of A
        self.B = np.array([0, 1/self.La]).reshape((2,1))
        self.C = np.array([[1, 0]], dtype=np.float)
        self.D = 0
        self.dist = np.array([[-1/self.J, 0]])         # Input Disturbance

        self.x = np.zeros((2,1))
        self.y = self.C.dot(self.x)
        self.T = T

    def output(self, t, x, u=0):
        dx = np.zeros((2,1), dtype=np.float)
        dx = self.A.dot(x.reshape(x.shape[0],1)) + self.B.dot(u) # + self.dist
        return dx.reshape(dx.shape[0])