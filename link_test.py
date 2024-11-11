import unittest
import sympy as sp
import numpy as np

from util import jacobian
from se3 import rot_x, trans
from link import Link, Model, _time_diff_matrix, Robot


class TestLink(unittest.TestCase):

    def test_time_diff_matrix(self):
        x, y = sp.symbols('x y')
        dx, dy = sp.symbols('dx dy')

        A = sp.Matrix([[x], [y]])
        dA = _time_diff_matrix(A, [x, y], [dx, dy])
        self.assertEqual(dA, sp.Matrix([[dx], [dy]]))

        B = sp.Matrix([[x, x**2], [y, y**2]])
        dB = _time_diff_matrix(B, [x, y], [dx, dy])
        self.assertEqual(dB, sp.Matrix([[dx, 2*x*dx], [dy, 2*y*dy]]))

        C = sp.Matrix([[x + y], [x*y]])
        dC = _time_diff_matrix(C, [x, y], [dx, dy])
        self.assertEqual(dC, sp.Matrix([[dx + dy], [x*dy + y*dx]]))


class TestSystem(unittest.TestCase):

    def test_pendulum(self):
        """
        simple pendulum with point mass at the end, length l, and angle θ
        mass m
        """
        l = 1
        m = 1
        g = 9.81

        theta, dtheta = sp.symbols('θ dθ')
        Tbn = rot_x(theta) @ trans(0, 0, -l)
        mass = Link(m, 0, np.zeros((3, 3)), np.zeros((6, 6)),
                    np.zeros((6, 6)), np.zeros((6, 6)))

        pendulum = Robot(links=[mass],
                         transforms=[Tbn],
                         params=[theta],
                         diff_params=[dtheta])

        model = pendulum.get_model(gvec=np.array([0, 0, -g]))
        self.assertEqual(model.M, sp.Matrix([[1]]))
        self.assertEqual(model.C, sp.Matrix([[0]]))
        self.assertEqual(model.D, sp.Matrix([[0]]))
        self.assertEqual(model.g, sp.Matrix([[g*sp.sin(theta)]]))

    def test_pendulum_com(self):
        """
        simple pendulum with point mass at the end, length l, and angle θ
        mass m. Center of mass adjusted by -l in z direction
        """
        l = 1
        m = 1
        g = 9.81

        theta, dtheta = sp.symbols('θ dθ')
        Tbn = rot_x(theta) @ trans(0, 0, -l)
        mass = Link(m, 0, np.zeros((3, 3)),
                    np.zeros((6, 6)), np.zeros((6, 6)), np.zeros((6, 6)),
                    center_of_mass=np.array([0, 0, -l]))

        pendulum = Robot(links=[mass],
                         transforms=[Tbn],
                         params=[theta],
                         diff_params=[dtheta])

        model = pendulum.get_model(gvec=np.array([0, 0, -g]))
        self.assertEqual(model.M, sp.Matrix([[4]])) # NB! 4 instead of 1 
        self.assertEqual(model.C, sp.Matrix([[0]]))
        self.assertEqual(model.D, sp.Matrix([[0]]))
        self.assertEqual(model.g, sp.Matrix([[2*g*sp.sin(theta)]])) # NB! 2
