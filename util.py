import numpy as np
from typing import Union
from numpy.typing import NDArray
from deprecated import deprecated

import sympy as sp
from sympy.matrices import MatrixBase

UtilFloat = Union[float, np.float64]


def is_symmetric(matrix: NDArray, atol: float = 1e-8) -> bool:
    """
    Check if a matrix is symmetric within a given tolerance.
    """
    return np.allclose(matrix, matrix.T, atol=atol)


def is_positive_definite(A: NDArray) -> bool:
    """
    Check if a matrix A is positive definite. If the matrix is not symmetric,
    check if A + A.T is positive definite.
    """
    if is_symmetric(A):
        return bool(np.all(np.linalg.eigvals(A) > 0))

    if A.shape[0] != A.shape[1]:
        return False

    return is_positive_definite(A + A.T)


def is_spd(A: NDArray) -> bool:
    """
    Check if a matrix A is symmetric positive definite.
    """
    return is_symmetric(A) and is_positive_definite(A)


def jacobian(x: MatrixBase, q: MatrixBase) -> MatrixBase:
    """
    Compute the Jacobian of a nx1 vector x with respect to a list of
    parameters q.
    """

    assert ((x.shape[0] == 1) or (x.shape[1] == 1))
    assert q.shape[1] == 1

    if (x.shape[0] == 1) and (x.shape[1] != 1):
        return _jacobian(x.T, q).T

    nx = x.shape[0]
    nq = q.shape[0]

    J = sp.zeros(nx, nq)
    for i in range(nx):
        J[i, :] = x[i].diff(q).T

    return J


@deprecated
def rot_x(angle: Union[float, np.float64]) -> NDArray:
    """
    Return the rotation matrix about the x-axis.
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])


@deprecated
def rot_y(angle: Union[float, np.float64]) -> NDArray:
    """
    Return the rotation matrix about the y-axis.
    """
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


@deprecated
def rot_z(angle: Union[float, np.float64]) -> NDArray:
    """
    Return the rotation matrix about the z-axis.
    """
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])


@deprecated
def is_rotation_matrix(matrix: NDArray, atol: float = 1e-8) -> bool:
    """
    Check if a matrix is a valid rotation matrix.
    """
    return np.allclose(np.linalg.det(matrix), 1, atol=atol) and np.allclose(matrix @ matrix.T, np.eye(3), atol=atol)
