import unittest
import sympy as sp
import numpy as np

from util import jacobian


class TestLink(unittest.TestCase):

    def test_jacobian(self):
        x, y, z = sp.symbols('x y z')

        q = sp.Matrix([x, y, z])

        m1 = sp.Matrix([[x]])
        dm1 = jacobian(m1, q)
        self.assertEqual(dm1, sp.Matrix([[1, 0, 0]]))

        m2 = sp.Matrix([[x + y**2 + x*z], [x*y*z]])
        dm2 = jacobian(m2, q)
        self.assertEqual(dm2, sp.Matrix([[1 + z, 2*y, x],
                                        [y*z, x*z, x*y]]))

        m3 = sp.Matrix([[0], [sp.sin(x)], [-sp.cos(x)]])
        qi = sp.Matrix([x])
        dm3 = jacobian(m3, qi)

        m4 = m3.T
        dm4 = jacobian(m4, qi)
        self.assertEqual(dm4, dm3.T)
