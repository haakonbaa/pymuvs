import numpy as np
from typing import Union
from numpy.typing import NDArray

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
        return np.all(np.linalg.eigvals(A) > 0)

    if A.shape[0] != A.shape[1]:
        return False

    return is_positive_definite(A + A.T)

def is_spd(A: NDArray) -> bool:
    """
    Check if a matrix A is symmetric positive definite.
    """
    return is_symmetric(A) and is_positive_definite(A)

def rot_x(angle: Union[float, np.float64]) -> NDArray:
    """
    Return the rotation matrix about the x-axis.
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])


def rot_y(angle: Union[float, np.float64]) -> NDArray:
    """
    Return the rotation matrix about the y-axis.
    """
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])

def rot_z(angle: Union[float, np.float64]) -> NDArray:
    """
    Return the rotation matrix about the z-axis.
    """
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

def is_rotation_matrix(matrix: NDArray, atol: float = 1e-8) -> bool:
    """
    Check if a matrix is a valid rotation matrix.
    """
    return np.allclose(np.linalg.det(matrix), 1, atol=atol) and np.allclose(matrix @ matrix.T, np.eye(3), atol=atol)
