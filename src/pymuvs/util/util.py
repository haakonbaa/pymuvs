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

    assert isinstance(x, sp.MatrixBase)
    assert isinstance(q, sp.MatrixBase) # TODO: list of symbols ok?
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

def time_diff_matrix(A: MatrixBase, q: list[sp.Symbol], dq: list[sp.Symbol]) -> MatrixBase:
    """
    Compute the time derivative of a matrix A. A is a function of q, and the
    time derivatives of q are given by dq.
    """
    assert len(q) == len(dq)

    Ad = sp.zeros(A.shape[0], A.shape[1])
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            Ad[r, c] = A[r, c].diff(sp.Matrix(q)).T @ sp.Matrix(dq)

    return Ad


def skew(x: Union[NDArray, MatrixBase]) -> Union[NDArray, MatrixBase]:
    if isinstance(x, sp.Matrix):
        return sp.Matrix([[0, -x[2], x[1]],
                          [x[2], 0, -x[0]],
                          [-x[1], x[0], 0]])
    if isinstance(x, np.ndarray):
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
