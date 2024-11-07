import sympy as sp
import numpy as np
from sympy.matrices import MatrixBase
from numpy import ndarray as NDArray
from typing_extensions import Self
from typing import TypeVar

_SIMPLIFY: bool = True


def set_simplify(simplify: bool):
    global _SIMPLIFY
    _SIMPLIFY = simplify


class SE3():
    """
    Represents an element of SE(3) - the special Euclidean group in 3D.
    https://en.wikipedia.org/wiki/Euclidean_group
    """

    def __init__(self) -> None:
        self._rotation: MatrixBase = sp.eye(3)
        self._translation: MatrixBase = sp.zeros(3, 1)

    def __repr__(self) -> str:
        return f"SE3(rotation={self._rotation}, translation={self._translation})"

    def __matmul__(self, other: 'SE3') -> 'SE3':
        """
        Multiply two elements of SE(3) together.

        Corresponds to applying the first transformation (other) and then the
        second transformation (self) to a point/vector.
        """
        if not isinstance(other, SE3):
            raise ValueError("Invalid type for multiplication.")

        res = SE3()
        res._rotation = self._rotation @ other._rotation
        res._translation = self._rotation @ other._translation + self._translation
        if _SIMPLIFY:
            res._rotation = sp.simplify(res._rotation)
            res._translation = sp.simplify(res._translation)
        return res

    def apply(self, point: MatrixBase | NDArray) -> MatrixBase:
        """
        Applies the transformation to a point or vector.

            [R, t]
        For [0, 1] in SE(3), applies the transformation
            R @ p + t
        to a point p:
        """
        assert isinstance(point, sp.Matrix) or isinstance(point, np.ndarray)
        assert point.shape[0] == 3
        point = sp.Matrix(point)
        expression: MatrixBase = self._rotation @ point + self._translation
        if _SIMPLIFY:
            return sp.simplify(expression)
        return expression

    def copy(self) -> 'SE3':
        """
        Create a deep copy of the SE3 element.
        """
        r: SE3 = SE3()
        r._rotation = self._rotation.copy()
        r._translation = self._translation.copy()
        return r


def rot_x(theta: sp.Symbol) -> SE3:
    """
    Create an element of SE(3) that represents a rotation around the x-axis.
    Resulting rotation will move a point around the x-axis by theta radians
    following the right-hand rule.
    """
    c = sp.cos(theta)
    s = sp.sin(theta)
    r = SE3()
    r._rotation = sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    return r


def rot_y(theta: sp.Symbol) -> SE3:
    """
    Create an element of SE(3) that represents a rotation around the y-axis.
    Resulting rotation will move a point around the y-axis by theta radians
    following the right-hand rule.
    """
    c = sp.cos(theta)
    s = sp.sin(theta)
    r = SE3()
    r._rotation = sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return r


def rot_z(theta: sp.Symbol) -> SE3:
    """
    Create an element of SE(3) that represents a rotation around the z-axis.
    Resulting rotation will move a point around the z-axis by theta radians
    following the right-hand rule.
    """
    c = sp.cos(theta)
    s = sp.sin(theta)
    r = SE3()
    r._rotation = sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return r


def trans(x: sp.Symbol, y: sp.Symbol, z: sp.Symbol) -> SE3:
    """
    Create an element of SE(3) that represents a translation.
    Resulting translation will move a point by x, y, z units in the x, y, z
    directions respectively.
    """

    t = SE3()
    t._translation = sp.Matrix([x, y, z])
    return t


def inv(T: SE3) -> SE3:
    """
    Invert an element of SE(3).
    Returns a new element of SE(3) that is the inverse of the input.
    T @ inv(T) = 0 (Applying 0 does not change the point)
    """

    r = T.copy()
    r._rotation = r._rotation.T
    r._translation = -r._rotation @ r._translation
    return r
