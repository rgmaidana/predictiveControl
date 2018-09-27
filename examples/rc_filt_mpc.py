#!/usr/bin/env python

import sys

class RCFilter:
    def __init__(self, R=10e3, C=10e-6, T=0.005, **kwargs):
        # Constructive parameters
        self.R = R
        self.Cap = C

        # Continuous-time state-space
        self.A = np.array([[-1/(self.R*self.Cap)]], dtype=np.float)
        self.B = np.array([[1]], dtype=np.float)
        self.C = np.array([[1/(self.R*self.Cap)]], dtype=np.float)

        self.T = T
        self.x = np.zeros((self.A.shape[1],1), dtype=np.float)
        self.u = np.zeros((self.B.shape[1],1), dtype=np.float)

    def output(self, t, x, u=0):
        dx = self.A.dot(x.reshape(self.x.shape)) + self.B.dot(u.reshape(self.u.shape))
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
    filt = RCFilter(T=0.01)
    
    # Instantiate MPC with RC filter model
    mpc = MPC(filt.A, filt.B, filt.C, T=filt.T)
    mpc.set_predict_horizon(15)         # Set prediction horizon
    mpc.set_control_horizon(2)          # Set control horizon
    mpc.dumin, mpc.dumax = np.array([-10]), np.array([10])      # Set restrictions to actuator variation and amplitude
    mpc.umin, mpc.umax = np.array([0]), np.array([100])         
    mpc.set_reference([10])               # Set reference (Volts)

    # Setup Nonstiff Ordinary Diff. Equation (ODE) solver (equivalent to matlab's ODE45)
    dt = 1e-3       # ODE derivation time
    solv = ode(filt.output).set_integrator('dopri5', method='rtol')

    x = np.array([[0,0]], dtype=np.float)
    while True:
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
        x = np.c_[x, mpc.x[:,-1]]

        # Append time
        mpc.t = np.append(mpc.t, mpc.t[-1]+mpc.T)
        if mpc.t[-1] >= 6:     # If timeout, break loop
            break

    # Print results
    print "\nSimulation finished\n"
    print "Reference: %.2f V" % mpc.get_reference()
    print "Final states at time %.2f seconds:" % mpc.t[-1]
    print "\tVoltage: \t%.2f V" % mpc.x[0,-1]
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(mpc.t, np.ones(mpc.t.shape)*mpc.get_reference(), 'k--', lw=2.0)
        plt.plot(mpc.t, x[0,:], 'b-', lw=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend(['Reference', 'Output'])
        plt.grid()
        plt.show()
    except ImportError:
        pass