import sympy as sp
#from ..util import jacobian
from ..util import time_diff_matrix

class Task:
    def __init__(self):
        self.f : sp.Matrix = None
        self.J : sp.Matrix = None
        self.dJ : sp.Matrix = None
        self.name : str = None

class TaskDesired:
    def __init__(self):
        self.sigma : sp.Matrix = None
        self.dsigma : sp.Matrix = None
        self.ddsigma : sp.Matrix = None
        self.name : str = None

def fn_to_task(fn : sp.Matrix, args : list[sp.Symbol], dargs : list[sp.Symbol], name : str = 'noname') -> Task:
    if isinstance(fn, sp.Expr):
        fn = sp.Matrix([fn])
    assert fn.shape[1] == 1
    assert fn.shape[0] > 0

    assert isinstance(args, list)
    assert len(args) > 0
    for a in args:
        assert isinstance(a, sp.Symbol)

    assert isinstance(dargs, list)
    assert len(dargs) == len(args)
    for a in dargs:
        assert isinstance(a, sp.Symbol)
    assert isinstance(name, str)

    task = Task()
    task.f = fn
    task.J = fn.jacobian(args)
    task.dJ = time_diff_matrix(task.J, args, dargs)
    task.name = name

    return task

def fn_to_task_desired(fn : sp.Matrix, t : sp.Symbol, name : str = 'noname') -> TaskDesired:
    if isinstance(fn, sp.Expr):
        fn = sp.Matrix([fn])
    assert fn.shape[1] == 1
    assert fn.shape[0] > 0
    assert isinstance(t, sp.Symbol)
    pass

    task = TaskDesired()
    task.sigma = fn
    task.dsigma = fn.diff(t)
    task.ddsigma = task.dsigma.diff(t)
    task.name = name

    return task
