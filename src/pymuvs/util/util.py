import numpy as np
from typing import Union
from numpy.typing import NDArray

import sympy as sp
from sympy.matrices import MatrixBase

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
        return jacobian(x.T, q).T

    nx = x.shape[0]
    nq = q.shape[0]

    J = sp.zeros(nx, nq)
    for i in range(nx):
        J[i, :] = x[i].diff(q).T

    return J
