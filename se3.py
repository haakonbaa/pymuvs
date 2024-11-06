import sympy as sp
import numpy as np

_SIMPLIFY = True

def set_simplify(simplify: bool):
    global _SIMPLIFY
    _SIMPLIFY = simplify


class SE3():
    def __init__(self):
        self._rotation = sp.eye(3)
        self._translation = sp.zeros(3, 1)

    def __repr__(self):
        return f"SE3(rotation={self._rotation}, translation={self._translation})"

    def __matmul__(self, other):
        if not type(other) == type(self):
            raise ValueError("Invalid type for multiplication.")

        res = SE3()
        res._rotation = self._rotation @ other._rotation
        res._translation = self._rotation @ other._translation + self._translation
        return res

    def apply(self, point: sp.Matrix) -> sp.Matrix:
        assert isinstance(point, sp.Matrix) or isinstance(point, np.ndarray)
        point = sp.Matrix(point)

        return self._rotation @ point + self._translation

    def copy(self):
        r = SE3()
        r._rotation = self._rotation.copy()
        r._translation = self._translation.copy()
        return r

def rot_x(theta: sp.Symbol) -> SE3:
    c = sp.cos(theta)
    s = sp.sin(theta)
    r = SE3()
    r._rotation = sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    return r

def rot_y(theta: sp.Symbol) -> SE3:
    c = sp.cos(theta)
    s = sp.sin(theta)
    r = SE3()
    r._rotation = sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return r

def rot_z(theta: sp.Symbol) -> SE3:
    c = sp.cos(theta)
    s = sp.sin(theta)
    r = SE3()
    r._rotation = sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return r

def trans(x: sp.Symbol, y: sp.Symbol, z: sp.Symbol) -> SE3:
    t = SE3()
    t._translation = sp.Matrix([x, y, z])
    return t

def inv(T: SE3) -> SE3:
    r = T.copy()
    r._rotation = r._rotation.T
    r._translation = -r._rotation @ r._translation
    return r
