import unittest
import sympy as sp
import numpy as np

from src.pymuvs.util import jacobian
from src.pymuvs.se3 import rot_x, rot_y, rot_z, trans
from src.pymuvs import Link, Model, Robot, Wrench
from src.pymuvs.link import _time_diff_matrix


class TestLink(unittest.TestCase):

    def test_tau_forces(self):

        f1, f2 = sp.symbols('f1 f2', real=True)
        force_f1 = Wrench(
            position=np.array([1, 0, 0]),
            wrench=sp.Matrix([0, f1, 0, 0, 0, 0]),
        )
        force_f2 = Wrench(
            position=np.array([-1, 0, 0]),
            wrench=sp.Matrix([0, -f2, 0, 0, 0, 0]),
        )

        # l1 is a sphere in three dimensions
        l1 = Link(mass=1,
                  volume=1,
                  inertia=np.eye(3)*(2/5)*1*1**2,
                  added_mass=np.zeros((6, 6)),
                  linear_damping=np.zeros((6, 6)),
                  quadratic_damping=np.zeros((6, 6)),
                  center_of_mass=np.array([0, 0, 0]),
                  center_of_buoyancy=np.zeros((3,)),
                  wrenches=[force_f1, force_f2],
                  )

        xn, yn, zn = sp.symbols('xn yn zn', real=True)
        phi, theta, psi = sp.symbols('ϕ θ ψ', real=True)

        dxn, dyn, dzn = sp.symbols('dxn dyn dzn', real=True)
        dphi, dtheta, dpsi = sp.symbols('dϕ dθ dψ', real=True)

        q = [xn, yn, zn, phi, theta, psi]
        dq = [dxn, dyn, dzn, dphi, dtheta, dpsi]

        Tbn = trans(xn, yn, zn) @ rot_z(psi) @ rot_y(theta) @ rot_x(phi)

        sphere = Robot(
            links=[l1],
            transforms=[Tbn],
            params=q,
            diff_params=dq,
            inputs=[f1, f2],
        )

        model = sphere.get_model(simplify=False,
                                 gvec=np.array([0, 0, -9.81]),
                                 bvec=np.array([0, 0, 9.81]),
                                 )
        qval = np.array([0, 0, 0, 0, 0, 0])
        dqval = np.array([0, 0, 0, 0, 0, 0])

        zval = np.array([1, 1])
        ddq_spin = model.eval(qval, dqval, zval)
        self.assertTrue(np.allclose(ddq_spin, np.array([0, 0, 0, 0, 0, 5])))

        zval = np.array([1, -1])
        ddq_trans = model.eval(qval, dqval, zval)
        self.assertTrue(np.allclose(ddq_trans, np.array([0, 2, 0, 0, 0, 0])))

        qval = np.array([sp.pi/8, 0, 0, 0, 0, 0])
        zval = np.array([1, 1])
        ddq_trans = model.eval(qval, dqval, zval)
        self.assertTrue(np.allclose(ddq_trans, np.array([0, 0, 0, 0, 0, 5])))
