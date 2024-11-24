__version__ = '0.0.1'
__author__ = 'Håkon Bårsaune'
__license__ = 'MIT'
__description__ = 'A package for generating mathematical models of underwater vehicles'

from .link import Link, Robot, Model, Wrench
from .robot import RobotV2

__all__ = ['Link', 'Robot', 'Model', 'RobotV2', 'Wrench']
