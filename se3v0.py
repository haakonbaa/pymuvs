import numpy as np
import sympy as sp
#from sympy import Matrix
from numpy.typing import NDArray
from typing import Optional, Union


from util import rot_x, rot_y, rot_z, is_rotation_matrix

_SIMPLIFY = True

def set_simplify(simplify: bool):
    global _SIMPLIFY
    _SIMPLIFY = simplify

class SE3():
    def __init__(self,
                 rotation: Optional[NDArray[np.float64]] = None,
                 translation: Optional[NDArray[np.float64]] = None
                 ):
        """
        Initialize the SE(3) object with a rotation matrix and translation vector.
        The rotation can be either a rotation matrix or a set of Euler angles.
        """

        # Validate Rotation

        if rotation is None:
            rotation = np.eye(3)

        assert (type(rotation) == np.ndarray or type(rotation) == sp.Matrix)

        rotation_verified = False

        if rotation.shape == (3, 3):
            assert is_rotation_matrix(rotation)
            rotation_verified = True

        if rotation.shape == (3,):
            rotation = rot_z(
                rotation[2]) @ rot_y(rotation[1]) @ rot_x(rotation[0])
            rotation_verified = True

        if not rotation_verified:
            raise ValueError("Invalid rotation matrix or Euler angles.")

        # Validate Translation

        if translation is None:
            translation = np.zeros(3)
        assert type(translation) == np.ndarray
        assert translation.shape == (3,)

        self._rotation: sp.Matrix = sp.Matrix(rotation)
        self._translation: sp.Matrix = sp.Matrix(translation)

    # ----- Getters and some operations -----

    def get_rotation_matrix(self) -> sp.Matrix:
        return self._rotation.copy()

    def get_translation_vector(self) -> sp.Matrix:
        return self._translation.copy()

    def copy(self):  # -> Self (python 3.11)
        r =  SE3()
        r._rotation = self._rotation.copy()
        r._translation = self._translation.copy()
        return r

    def __repr__(self) -> str:
        return f"SE3(rotation={self._rotation}, translation={self._translation})"

    # ----- SE(3) Operations -----
    # NOTE! These operations are relative to the SE(3) frame, i.e.
    # translating (1, 0, 0) will move the object along the x-axis of the SE(3)
    # frame, not the world frame.

    def rotate_x(self, angle: Union[float, sp.Symbol]):
        rotmat = sp.Matrix([
            [1, 0, 0],
            [0, sp.cos(angle), -sp.sin(angle)],
            [0, sp.sin(angle), sp.cos(angle)]
        ])
        self._rotation = rotmat @ self._rotation
        if _SIMPLIFY:
            self._rotation = sp.simplify(self._rotation)
        return self

    def rotate_y(self, angle: Union[float, sp.Symbol]):
        rotmat = sp.Matrix([
            [sp.cos(angle), 0, sp.sin(angle)],
            [0, 1, 0],
            [-sp.sin(angle), 0, sp.cos(angle)]
        ])
        self._rotation = rotmat @ self._rotation
        if _SIMPLIFY:
            self._rotation = sp.simplify(self._rotation)
        return self

    def rotate_z(self, angle: Union[float, sp.Symbol]):
        rotmat = sp.Matrix([
            [sp.cos(angle), -sp.sin(angle), 0],
            [sp.sin(angle), sp.cos(angle), 0],
            [0, 0, 1]
        ])
        self._rotation = rotmat @ self._rotation
        if _SIMPLIFY:
            self._rotation = sp.simplify(self._rotation)
        return self

    def translate(self, x: Union[float, sp.Symbol], y: Union[float, sp.Symbol], z: Union[float, sp.Symbol]):
        """
        translates the SE(3) object by the given x, y, z values in the SE(3) frame.
        """
        self._translation += self._rotation @ sp.Matrix([x, y, z])
        return self

    def translate_world_frame(self, x: Union[float, sp.Symbol], y: Union[float, sp.Symbol], z: Union[float, sp.Symbol]):
        """
        translates the SE(3) object by the given x, y, z values in the world frame.
        """
        self._translation += sp.Matrix([x, y, z])
        if _SIMPLIFY:
            self._translation = sp.simplify(self._translation)
        return self

    def inv(self):
        self._translation = -self._rotation.T @ self._translation
        self._rotation = self._rotation.T
        return self

    def apply(self, vector: NDArray[np.float64]):
        """
        Applies the SE(3) transformation to the given vector.
        """
        assert (isinstance(vector, np.ndarray) or isinstance(vector, sp.Matrix))
        assert vector.shape == (3,)

        return self._rotation @ sp.Matrix(vector) + self._translation

