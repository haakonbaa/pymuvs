import unittest
import sympy as sp
import numpy as np

from se3 import trans, rot_x
from link import _jacobian, _time_diff_matrix, Link, Robot

class TestLink(unittest.TestCase):

    def test_jacobian(self):
        x, y, z = sp.symbols('x y z')

        q = sp.Matrix([x, y, z])

        m1 = sp.Matrix([[x]])
        dm1 = _jacobian(m1, q)
        self.assertEqual(dm1, sp.Matrix([[1, 0, 0]]))

        m2 = sp.Matrix([[x + y**2 + x*z], [x*y*z]])
        dm2 = _jacobian(m2, q)
        self.assertEqual(dm2, sp.Matrix([[1 + z, 2*y, x],
                                        [y*z, x*z, x*y]]))

        m3 = sp.Matrix([[0], [sp.sin(x)], [-sp.cos(x)]])
        qi = sp.Matrix([x])
        dm3 = _jacobian(m3, qi)


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
        mass = Link(m, 0, np.zeros((3,3)), np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6)))

        pendulum = Robot(links = [mass],
                         transforms = [Tbn],
                         params = [theta],
                         diff_params = [dtheta])

        model = pendulum.get_model(gvec = np.array([0, 0, -g]))
        self.assertEqual(model.M, sp.Matrix([[1]]))
        self.assertEqual(model.C, sp.Matrix([[0]]))
        self.assertEqual(model.D, sp.Matrix([[0]]))
        self.assertEqual(model.g, sp.Matrix([[g*sp.sin(theta)]]))
