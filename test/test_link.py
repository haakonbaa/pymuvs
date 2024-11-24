import unittest
import sympy as sp
import numpy as np

from src.pymuvs.util import jacobian
from src.pymuvs.se3 import rot_x, trans
from src.pymuvs import Link, Model, Robot
from src.pymuvs.link import _time_diff_matrix, _Bu_to_B_and_u


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
        self.assertEqual(model.M, sp.Matrix([[4]]))  # NB! 4 instead of 1
        self.assertEqual(model.C, sp.Matrix([[0]]))
        self.assertEqual(model.D, sp.Matrix([[0]]))
        self.assertEqual(model.g, sp.Matrix([[2*g*sp.sin(theta)]]))  # NB! 2

    def test_pendulum_damping(self):
        """
        simple pendulum with point mass at the end, length l, and angle θ
        mass m. Added damping term
        """
        l = 1
        m = 1
        g = 9.81
        dy = 2

        theta, dtheta = sp.symbols('θ dθ')
        Tbn = rot_x(theta) @ trans(0, 0, -l)
        Damping = np.diag([0, dy, 0, 0, 0, 0])
        mass = Link(m, 0, np.zeros((3, 3)),
                    np.zeros((6, 6)), Damping, np.zeros((6, 6)),
                    center_of_mass=np.array([0, 0, 0]))

        pendulum = Robot(links=[mass],
                         transforms=[Tbn],
                         params=[theta],
                         diff_params=[dtheta])

        model = pendulum.get_model(gvec=np.array([0, 0, -g]))
        self.assertEqual(model.M, sp.Matrix([[1]]))
        self.assertEqual(model.C, sp.Matrix([[0]]))
        self.assertEqual(model.D, sp.Matrix([[dy]]))
        self.assertEqual(model.g, sp.Matrix([[g*sp.sin(theta)]]))

    def test_double_pendulum(self):
        """
        double pendulum with point mass at the end, length l, and angles θ1, θ2
        masses m1, m2
        """
        l1 = 1
        l2 = 1
        m1 = 1
        m2 = 1
        g = 9.81

        theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2 = sp.symbols(
            'θ1 θ2 dθ1 dθ2 ddθ1 ddθ2')
        Tl1n = rot_x(theta1) @ trans(0, 0, -l1)
        Tl2n = Tl1n @ rot_x(-theta1) @ rot_x(theta2) @ trans(0, 0, -l2)

        l1 = Link(m1, 0, np.zeros((3, 3)), np.zeros((6, 6)),
                  np.zeros((6, 6)), np.zeros((6, 6)))
        l2 = Link(m2, 0, np.zeros((3, 3)), np.zeros((6, 6)),
                  np.zeros((6, 6)), np.zeros((6, 6)))
        double_pendulum = Robot(links=[l1, l2],
                                transforms=[Tl1n, Tl2n],
                                params=[theta1, theta2],
                                diff_params=[dtheta1, dtheta2])
        model = double_pendulum.get_model(gvec=np.array([0, 0, -g]))

        dq = sp.Matrix([[dtheta1], [dtheta2]])
        ddq = sp.Matrix([[ddtheta1], [ddtheta2]])
        m = model.M @ ddq + model.C @ dq + model.D @ dq + model.g
        # TODO: verify that this is correct?
        print(f"{model.M=}")
        print(f"{model.C=}")
        print(f"{model.D=}")
        print(f"{model.g=}")
        print(f"{model.J=}")


class TestPrivateFunctions(unittest.TestCase):

    def test_Bu_to_B_and_u(self):
        """
        test the function Bu_to_B_and_u
        """
        x, y, z = sp.symbols('x y z')
        Bu = sp.Matrix([[x], [y], [z]])
        B, u = _Bu_to_B_and_u(Bu)
        self.assertEqual(B, sp.eye(3))
        self.assertEqual(u, sp.Matrix([[x], [y], [z]]))

        Bu = sp.Matrix([x + y, x - y, z])
        B, u = _Bu_to_B_and_u(Bu)
        self.assertEqual(Bu, B @ u)
        self.assertEqual(B.free_symbols, set())
        self.assertEqual(u.shape, (3, 1))

        Bu = sp.Matrix([x + y + 1, x - y, z])
        B, u = _Bu_to_B_and_u(Bu)
        self.assertEqual(Bu, B @ u)
        self.assertEqual(B.free_symbols, set())
        self.assertEqual(u.shape, (4, 1))

        Bu = sp.zeros(3, 1)
        B, u = _Bu_to_B_and_u(Bu)
        print(f"{Bu=}")
        print(f"{B=}")
        print(f"{u=}")
