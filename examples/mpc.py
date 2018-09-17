#!/usr/bin/env python

from DCMotor import Motor
import numpy as np
import sys

if __name__ == '__main__':
    # Try importing predictiveControl package
    try:
        from predictivecontrol import MPC
    except ImportError:
        print "\nPredictive control package not installed."
        print "To install, go to the root folder of this repository and run \"sudo pip install -e .\"\n"
        sys.exit(0)

    # Try importing the ODE solver
    try:
        from scipy.integrate import ode
    except ImportError:
        print "\nThis simulation depends on the ODE solver from the scipy package."
        print "To install, run \"sudo pip install -U scipy\"\n"
        sys.exit(0)

    # Instantiate DC Motor model (sampling time of 0.05 seconds)
    motor = Motor(T=0.05)
    
    # Instantiate MPC with DC motor model
    mpc = MPC(motor)
    mpc.set_predict_horizon(15)         # Set prediction horizon
    mpc.set_control_horizon(4)          # Set control horizon
    mpc.umin, mpc.umax = -1, 6          # Set actuation limits
    mpc.dumin, mpc.dumax = -0.5, 1.5
    mpc.set_reference(10)               # Set reference (rad/s)
    
    # Setup Nonstiff Ordinary Diff. Equation (ODE) solver (equivalent to matlab's ODE45)
    dt = 1e-3       # ODE derivation time
    solv = ode(motor.output).set_integrator('dopri5', method='rtol')   

    # Run for some seconds
    timeout = 4
    while True:
        # Run MPC (will update controlled input u)
        mpc.run()

        # Solve ODE (simulate motor based on model)
        solv.set_initial_value(mpc.x0)
        solv.set_f_params(mpc.u[-1])        # Apply control input into system
        while solv.successful() and solv.t < mpc.T:
            solv.integrate(solv.t+dt)

        # Update states (equivalent to sensing)
        mpc.x0 = solv.y.reshape((2,1))
        mpc.x = np.c_[mpc.x, mpc.x0]

        # Append time
        mpc.t = np.append(mpc.t, mpc.t[-1]+mpc.T)
        if mpc.t[-1] >= timeout:     # If timeout, break loop
            break

    # Print results
    print "\nSimulation finished\n"
    print "Reference: %.2f rad/s" % mpc.get_reference()
    print "Final states at time %.2f seconds:" % mpc.t[-1]
    print "\tAngular velocity: \t%.2f rad/s" % mpc.x[0,-1]
    print "\tAngular acceleration: \t%.2f rad/s^2" % mpc.x[1,-1]

    # Plot results
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(mpc.t, np.ones(mpc.t.shape)*mpc.get_reference(), 'k--', lw=2.0)
        plt.plot(mpc.t, mpc.x[0,:], 'b-', lw=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('Angular velocity (rad/s)')
        plt.legend(['Reference', 'Output'])
        plt.grid()
        plt.show()
    except ImportError:
        pass