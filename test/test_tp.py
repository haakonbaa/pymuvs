import unittest
import sympy as sp
import numpy as np

from src.pymuvs.tp import Task, fn_to_task, TaskDesired, fn_to_task_desired
from src.pymuvs.codegen import tasks_to_cpp


class TestTP(unittest.TestCase):

    def test_tp_Task(self):
        x, y = sp.symbols('x y')
        dx, dy = sp.symbols('dx dy')
        f = x + y
        try:
            task1 = fn_to_task(sp.Matrix([f]), [x, y], [dx, dy])
            task2 = fn_to_task(f, [x, y], [dx, dy])
        except AssertionError as e:
            self.fail(e)

        f = sp.Matrix([x, y])
        task1 = fn_to_task(f, [x, y], [dx, dy])
        self.assertEqual(task1.f == f, True)
        self.assertEqual(task1.J == sp.eye(2), True)
        self.assertEqual(task1.dJ == sp.zeros(2, 2), True)

        f = sp.Matrix([x**2*y**2, x**2+y**2])
        task = fn_to_task(f, [x, y], [dx, dy])
        self.assertEqual(task.f == f, True)
        self.assertEqual(task.J == sp.Matrix([
            [2*x*y**2, 2*x**2*y],
            [2*x, 2*y]]), True)
        self.assertEqual(task.dJ == sp.Matrix([
            [2*dx*y**2+4*x*y*dy, 2*dy*x**2+4*x*dx*y],
            [2*dx, 2*dy]]), True)

    def test_tp_TaskDesired(self):

        t = sp.symbols('t')
        f = sp.Matrix([sp.sin(t), sp.cos(t)])

        task_d = fn_to_task_desired(f, t)
        self.assertEqual(task_d.sigma == f, True)
        self.assertEqual(task_d.dsigma == sp.Matrix([sp.cos(t),-sp.sin(t)]), True)
        self.assertEqual(task_d.ddsigma == sp.Matrix([-sp.sin(t),-sp.cos(t)]), True)

    def test_tp_codegen(self):
        t, x, y, dx, dy = sp.symbols('t x y dx dy')
        f = sp.Matrix([x, y])
        sigma = sp.Matrix([sp.sin(t), sp.cos(t)])

        q = [x, y]
        dq = [dx, dy]
        header, code = tasks_to_cpp([
            fn_to_task(f, [x, y], [dx, dy], name = 'endEffectorPos'),
            fn_to_task_desired(sigma, t, name = 'endEffectorCircle'),
        ], t, q, dq)

        with open('tasks.cpp', 'w') as f:
            f.write(code)
        with open('tasks.h', 'w') as f:
            f.write(header)


