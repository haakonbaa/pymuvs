"""
simple pendulum with point mass at the end, length l, and angle θ and mass m
"""
import sympy as sp
import numpy as np

from pymuvs import Link, Robot
from pymuvs.se3 import rot_x, trans

l = 1 # length of pendulum
m = 1 # mass of pendulum
g = 9.81 # gravity

theta, dtheta = sp.symbols('θ dθ')
Tbn = rot_x(theta) @ trans(0, 0, -l) # Transform from link- to inertial-frame
end_mass = Link(m, 0, np.zeros((3, 3)), np.zeros((6, 6)), np.zeros((6, 6)),
        np.zeros((6, 6)))

# Robot has links, each of which has their own transform
pendulum = Robot(links=[end_mass],
    transforms=[Tbn],
    params=[theta],
    diff_params=[dtheta])

model = pendulum.get_model(gvec=np.array([0, 0, -g])) # gravity in -z direction
print(f"{model.M = }")
print(f"{model.C = }")
print(f"{model.D = }")
print(f"{model.g = }")
