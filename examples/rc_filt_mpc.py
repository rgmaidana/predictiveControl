#!/usr/bin/env python

import sys

class RCFilter:
    def __init__(self, R=10e3, C=10e-6, T=0.005, **kwargs):
        # Constructive parameters
        self.R = R
        self.Cap = C

        # Continuous-time state-space
        self.A = np.array([[-1/(self.R*self.Cap)]], dtype=np.float)
        self.B = np.array([[1/(self.R*self.Cap)]], dtype=np.float)
        self.C = np.array([[1]], dtype=np.float)
        self.D = np.zeros((self.C.shape[0],self.B.shape[1]))

        self.T = T
        self.x = np.zeros((self.A.shape[1],1), dtype=np.float)
        self.u = np.zeros((self.B.shape[1],1), dtype=np.float)
        self.y = np.zeros((self.C.shape[0],1), dtype=np.float)

    def output(self, t, x, u=0):
        dx = self.A.dot(x.reshape(self.x.shape)) + self.B.dot(u.reshape(self.u.shape))
        self.y = self.C.dot(x.reshape(self.x.shape)) + self.D.dot(u.reshape(self.u.shape))
        return dx

if __name__ == "__main__":
    # Try importing predictivecontrol package
    try:
        from predictivecontrol import MPC
    except ImportError:
        print "\nPredictive control, scipy or numpy packages not installed."
        print "To install, go to the root folder of this repository and run \"pip install -e .\""
        print "The predictivecontrol package will automatically install scipy and numpy.\n"
        sys.exit(0)

    # Try importing the ODE solver
    try:
        from scipy.integrate import ode
    except ImportError:
        print "\nThis simulation depends on the ODE solver from the scipy package."
        print "To install, run \"pip install -U scipy\"\n"
        sys.exit(0)

    # Try importing numpy
    try:
        import numpy as np
    except ImportError:
        print "\nThis simulation depends on the numpy package."
        print "To install, run \"pip install -U numpy\"\n"
        sys.exit(0)

    # Instantiate RC filter model (sampling time of 0.01 seconds)
    filt = RCFilter(T=1e-2)
    
    # Instantiate MPC with RC filter model
    mpc = MPC(filt.A, filt.B, filt.C, T=filt.T)
    mpc.set_predict_horizon(15)         # Set prediction horizon
    mpc.set_control_horizon(2)          # Set control horizon
    mpc.dumin, mpc.dumax = np.array([-10]), np.array([10])      # Set restrictions to actuator variation and amplitude
    mpc.umin, mpc.umax = np.array([0]), np.array([100])         
    mpc.set_reference(np.array([10]))               # Set reference (Volts)
    mpc.set_output_weights(np.array([1]))           # Set output weight

    # Setup Nonstiff Ordinary Diff. Equation (ODE) solver (equivalent to matlab's ODE45)
    dt = 1e-3       # ODE derivation time
    solv = ode(filt.output).set_integrator('dopri5', method='rtol')

    # Run for some seconds
    timeout = 0.5
    x = np.zeros((mpc.A.shape[0],2))
    u = np.zeros((mpc.B.shape[1],2))
    y = np.zeros((mpc.C.shape[0],2))
    while True:
        # Run MPC (will update controlled input u)
        mpc.run()

        # Solve ODE (simulate motor based on model)
        solv.set_initial_value(mpc.x[:,-1]) # Current initial value is last state
        solv.set_f_params(mpc.u[0,-1])        # Apply control input into system
        while solv.successful() and solv.t < mpc.T:
            solv.integrate(solv.t+dt)

        # Update states (equivalent to sensing)
        # Number of states kept by MPC are bound by prediction horizon, to avoid memory issues on continuous use
        mpc.x = np.roll(mpc.x, -1)
        mpc.x[:,-1] = solv.y
        mpc.y[:,-1] = mpc.C.dot(mpc.x[:,-1].reshape(mpc.x[:,-1].shape))
        x = np.c_[x, mpc.x[:,-1]]
        u = np.c_[u, mpc.u[:,-1]]
        y = np.c_[y, mpc.y[:,-1]]

        # Append time
        mpc.t = np.append(mpc.t, mpc.t[-1]+mpc.T)
        if mpc.t[-1] >= timeout:     # If timeout, break loop
            break

    # Print results
    print "\nSimulation finished\n"

    print "Setpoints:"
    for i in range(len(y[:,-1])):
        print "\tR%d: \t%.2f" % (i+1, mpc.get_reference()[i])
    
    print "\nFinal states at time %.2f seconds:" % mpc.t[-1]
    for i in range(len(x[:,-1])):
        print "\tx%d: \t%.2f" % (i+1, x[i,-1])

    print "\nOutputs at time %.2f seconds:" % mpc.t[-1]
    for i in range(len(y[:,-1])):
        print "\ty%d: \t%.2f" % (i+1, y[i,-1])

    print "\nSteady-state error:"
    for i in range(len(y[:,-1])):
        print "\ty%d: \t%.2f" % (i+1, mpc.get_reference()[i]-y[i,-1])

    # Plot results
    try:
        import matplotlib.pyplot as plt

        # Plot states
        plt.figure()
        for k in range(x.shape[0]):
            plt.plot(mpc.t, x[k,:], lw=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('x')
        legend = []
        for k in range(0,x.shape[0]):
            legend.append('x%d' % (k+1))
        plt.legend(legend)
        plt.grid()

        # Plot inputs
        plt.figure()
        for k in range(u.shape[0]):
            plt.plot(mpc.t, u[k,:], lw=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('x')
        legend = [0 for _ in range(u.shape[0]*2)]
        for k in range(u.shape[0]):
            legend[k] = 'u%d' % (k+1)
        plt.legend(legend)
        plt.grid()

        # Plot outputs
        plt.figure()
        for k in range(y.shape[0]):
            ax = plt.plot(mpc.t, np.ones(mpc.t.shape)*mpc.get_reference()[k], '--', lw=2.0)
            plt.plot(mpc.t, y[k,:], color=ax[0].get_color(), lw=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('x')
        legend = []
        for k in range(0,y.shape[0]):
            legend.append('Reference %d' % (k+1))
            legend.append('y%d' % (k+1))
        plt.legend(legend)
        plt.grid()

        # Show figures
        plt.show()
    except ImportError:
        pass