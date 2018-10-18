#!/usr/bin/env python

import sys

# We define a DCMotor class for convenience, and for using its output function in the ODE solver
class RotBody:
    def __init__(self, **kwargs):
        # Rotating body with 3 inputs and 3 outputs
        self.A = np.array([[-0.04545,  -0.02727,  -0.02727],
                           [-0.02727,  -0.04545,  -0.02727],
                           [-0.02727,  -0.02727,  -0.04545]], dtype=np.float)
        self.B = np.array([[0.2273,  0.1364,  0.1364],
                           [0.1364,  0.2273,  0.1364],
                           [0.1364,  0.1364,  0.2273]], dtype=np.float)
        self.C = np.eye(self.A.shape[0])
        
        self.x = np.zeros((self.A.shape[1],1), dtype=np.float)
        self.u = np.zeros((self.B.shape[1],1), dtype=np.float)
        
    def output(self, t, x, u=0):
        dx = self.A.dot(x.reshape(self.x.shape)) + self.B.dot(u.reshape(self.u.shape)) # + self.dist
        return dx

if __name__ == '__main__':
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


    # Instantiate model
    body = RotBody()
    
    # Instantiate MPC with model
    mpc = MPC(body.A, body.B, body.C, T=0.01, discretize=True)
    mpc.set_predict_horizon(50)         # Set prediction horizon
    mpc.set_control_horizon(4)          # Set control horizon
    mpc.dumin, mpc.dumax = np.array([-10,-10,-10]), np.array([10,10,10])    # Set restrictions to torque
    mpc.umin, mpc.umax = np.array([-100,-100,-100]), np.array([100,100,100])          
    mpc.set_reference(np.array([10,10,10]))               # Set reference (rad/s)
    
    # Setup Nonstiff Ordinary Diff. Equation (ODE) solver (equivalent to matlab's ODE45)
    dt = 1e-3       # ODE derivation time
    solv = ode(body.output).set_integrator('dopri5', method='rtol')   

    # Run for some seconds
    timeout = 4
    x = mpc.x[:,0]
    x = np.c_[x, mpc.x[:,1]]
    while True:
        # Run MPC (will update controlled input u)
        mpc.run()

        # Solve ODE (simulate motor based on model)
        solv.set_initial_value(mpc.x[:,-1]) # Current initial value is last state
        solv.set_f_params(mpc.u[:,-1])        # Apply control input into system
        while solv.successful() and solv.t < mpc.T:
            solv.integrate(solv.t+dt)

        # Update states (equivalent to sensing)
        # Number of states kept by MPC are bound by prediction horizon, to avoid memory issues on continuous use
        mpc.x = np.roll(mpc.x[-mpc.x.shape[0]:,:], -1)
        mpc.x[:,-1] = solv.y
        x = np.c_[x, mpc.x[:,-1]]

        # Append time
        mpc.t = np.append(mpc.t, mpc.t[-1]+mpc.T)
        if mpc.t[-1] >= timeout:     # If timeout, break loop
            break

    # Print results
    print "\nSimulation finished\n"
    print "Reference: %.2f rad/s" % mpc.get_reference()[0]
    print "Final states at time %.2f seconds:" % mpc.t[-1]
    print "\tAngular velocity: \t%.2f rad/s" % mpc.x[0,-1]

    # Plot results
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(mpc.t, np.ones(mpc.t.shape)*mpc.get_reference()[0], 'k--', lw=2.0)
        plt.plot(mpc.t, x[0,:], 'b-', lw=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('Angular velocity (rad/s)')
        plt.legend(['Reference', 'Output'])
        plt.grid()
        plt.show()
    except ImportError:
        pass