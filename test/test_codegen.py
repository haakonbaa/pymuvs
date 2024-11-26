import unittest
import sympy as sp
import numpy as np

from src.pymuvs import Link, Model, Robot, Wrench
from src.pymuvs.se3 import rot_x, rot_y, rot_z, trans
from src.pymuvs.codegen import model_to_cpp, matrix_to_cppfn


class TestCodeGen(unittest.TestCase):

    def test_codegen_sphere_with_input(self):
        """
        code generation of a sphere with input forces
        """
        f1, f2 = sp.symbols('f1 f2', real=True)
        force_f1 = Wrench(
            position=np.array([1, 0, 0]),
            wrench=sp.Matrix([0, f1, 0, 0, 0, 0]),
        )
        force_f2 = Wrench(
            position=np.array([-1, 0, 0]),
            wrench=sp.Matrix([0, -f2, 0, 0, 0, 0]),
        )
        tau1 = sp.symbols('τ1', real=True)
        tau2 = sp.symbols('τ2', real=True)

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
            inputs=[f1, f2, tau1],
        )

        model = sphere.get_model(simplify=False,
                                 gvec=np.array([0, 0, -9.81]),
                                 bvec=np.array([0, 0, 9.81]),
                                 generalized_forces={phi: tau1},
                                 )

        code, header, body = model_to_cpp(model)
        with open('model.cpp', 'w') as f:
            f.write(body)

        with open('model.h', 'w') as f:
            f.write(header)
