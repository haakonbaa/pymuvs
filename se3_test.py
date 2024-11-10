import sympy as sp
import unittest

from se3 import *


class TestSE3(unittest.TestCase):
    r, p, y = sp.symbols('ϕ θ ψ')
    dr, dp, dy = sp.symbols('dϕ dθ dψ')
    T = rot_z(y) @ rot_y(p) @ rot_x(r)

    def test_SE3(self):
        # TODO: Implement test
        pass

    def test_rot_x(self):
        r = sp.symbols('ϕ')
        T = rot_x(r)

        R = T.get_rotation()

        Rpi = R.subs({r: sp.pi})
        self.assertTrue(Rpi == sp.Matrix([[1, 0, 0],
                                          [0, -1, 0],
                                          [0, 0, -1]]))

        Rpi2 = R.subs({r: sp.pi/2})
        self.assertTrue(Rpi2 == sp.Matrix([[1, 0, 0],
                                           [0, 0, -1],
                                           [0, 1, 0]]))

        R0 = R.subs({r: 0})
        self.assertTrue(R0 == sp.eye(3))

    def test_rot_y(self):
        p = sp.symbols('θ')
        T = rot_y(p)

        R = T.get_rotation()

        Rpi = R.subs({p: sp.pi})
        self.assertTrue(Rpi == sp.Matrix([[-1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, -1]]))

        Rpi2 = R.subs({p: sp.pi/2})
        self.assertTrue(Rpi2 == sp.Matrix([[0, 0, 1],
                                           [0, 1, 0],
                                           [-1, 0, 0]]))

        R0 = R.subs({p: 0})
        self.assertTrue(R0 == sp.eye(3))

    def test_get_jacobian(self):
        xn, yn, zn = sp.symbols('xn yn zn', real=True)
        phi, theta, psi = sp.symbols('ϕ θ ψ', real=True)
        Tbn = trans(xn, yn, zn) @ rot_z(psi) @ rot_y(theta) @ rot_x(phi)

        q = [xn, yn, zn, phi, theta, psi]
        J = Tbn.get_jacobian(q)

        # page 28 R^T Fossen (transposed) #TODO: Cite properly
        # and page 29 T^-1
        Je = sp.Matrix([[sp.cos(psi)*sp.cos(theta), sp.sin(psi)*sp.cos(theta), -sp.sin(theta), 0, 0, 0],
                        [-sp.sin(psi)*sp.cos(phi)+sp.cos(psi)*sp.sin(theta)*sp.sin(phi), sp.cos(psi)*sp.cos(
                            phi) + sp.sin(phi)*sp.sin(theta)*sp.sin(psi), sp.cos(theta)*sp.sin(phi), 0, 0, 0],
                        [sp.sin(psi)*sp.sin(phi) + sp.cos(psi)*sp.cos(phi)*sp.sin(theta), -sp.cos(psi)*sp.sin(
                            phi) + sp.sin(theta)*sp.sin(psi)*sp.cos(phi), sp.cos(theta)*sp.cos(phi), 0, 0, 0],
                        [0, 0, 0, 1, 0, -sp.sin(theta)],
                        [0, 0, 0, 0, sp.cos(phi), sp.sin(phi)*sp.cos(theta)],
                        [0, 0, 0, 0, -sp.sin(phi), sp.cos(phi)*sp.cos(theta)]])
        self.assertTrue(J == Je)

    def test_rot_z(self):
        y = sp.symbols('ψ')
        T = rot_z(y)

        R = T.get_rotation()

        Rpi = R.subs({y: sp.pi})
        self.assertTrue(Rpi == sp.Matrix([[-1, 0, 0],
                                          [0, -1, 0],
                                          [0, 0, 1]]))

        Rpi2 = R.subs({y: sp.pi/2})
        self.assertTrue(Rpi2 == sp.Matrix([[0, -1, 0],
                                           [1, 0, 0],
                                           [0, 0, 1]]))

        R0 = R.subs({y: 0})
        self.assertTrue(R0 == sp.eye(3))

    def test_skew(self):
        x, y, z = sp.symbols('x y z')
        v = sp.Matrix([x, y, z])
        S = skew(v)

        self.assertTrue(S == sp.Matrix([[0, -z, y],
                                        [z, 0, -x],
                                        [-y, x, 0]]))
        self.assertTrue(S.T == -S)

    def test_down_skew(self):
        x, y, z = sp.symbols('x y z')
        S = sp.Matrix([[0, -z, y],
                       [z, 0, -x],
                       [-y, x, 0]])
        v = down_skew(S)

        self.assertTrue(v == sp.Matrix([x, y, z]))

    def test_rotmat_to_angvel_matrix_frameb(self):
        r, p, y = sp.symbols('ϕ θ ψ')
        dr, dp, dy = sp.symbols('dϕ dθ dψ')
        T = rot_z(y) @ rot_y(p) @ rot_x(r)
        R = self.T.get_rotation()
        J = rotmat_to_angvel_matrix_frameb(R, [r, p, y])

        # Equation 2.41 in [2]

        ExpectedR = sp.Matrix([[1, 0, -sp.sin(p)],
                               [0, sp.cos(r), sp.sin(r)*sp.cos(p)],
                               [0, -sp.sin(r), sp.cos(r)*sp.cos(p)]])

        self.assertTrue(J == ExpectedR)
        self.assertTrue(J.subs({r: 0, p: 0, y: 0}) == sp.eye(3))
