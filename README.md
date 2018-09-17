# Predictive Control in Python

This package implements Predictive Control techniques in Python (v2.7). 
Currently it supports only Model-Predictive Control (MPC), although a class for Economic MPC has been added (not tested!).

## Dependencies

* [numpy](https://www.numpy.org/)
* [scipy](https://www.scipy.org/)

## Installation

Clone or download the repository and install with pip:

```
pip -e <path_to_repository>
```

## Usage

Simply import the class of controller wanted and instantiate it with a valid state-space model.
This model must be a class with the constructive matrices (i.e., A, B, C and D) as attributes.
Then use the ```run()``` method to update the controller output based on the last sensed or estimated states.

You may also set the prediction and control horizons, the actuation limits and the reference for your control system.

```
from DCMotor import Motor
from predictiveControl import MPC

dcmotor = Motor()
mpcontroller = MPC(dcmotor)

mpc.set_predict_horizon(10)
mpc.set_control_horizon(4)
mpc.umin, mpc.umax = 0, 100
mpc.dumin, mpc.dumax = -0.5, 0.5

mpc.set_reference(10)

mpc.run()
```

## Examples

A full example of voltage control in a simulated DC Motor can be found in the [examples]() folder.
The motor is modelled as a Python class, and its constructive parameters can be accessed as class attributes.
You can run it with:

```python mpc.py```

If you want to plot the results, you must install the matplotlib package:

```pip install matplotlib```

## To-Do

* Test MPC with a different model
* Test E-MPC with DC Motor and another model
* Test both controllers with MIMO models
* Account for external disturbances in the controllers

## Credits
Code inspired by [Prof. Dr. Aurelio Salton](https://scholar.google.com/citations?user=uyWSHmAAAAAJ&hl=en)'s MPC matlab example.

## Collaborators

* [Renan Maidana](https://github.com/rgmaidana)
