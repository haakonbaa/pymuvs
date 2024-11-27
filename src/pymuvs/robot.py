
import numpy as np
from numpy.typing import NDArray
from deprecated import deprecated

from .link import Link
from .se3 import SE3


class RobotV2():

    @deprecated
    def __init__(self, link: Link, transform: SE3):
        # each link has a unique id
        self._link_to_id = {link: 0}
        self._parent = {0: None}
        # storage of links and transforms by id
        self._links = [link]
        self._transforms = [transform]

    def attach(self, link: Link, transform: SE3, parent: Link):
        link_id = len(self._links)
        self._links.append(link)
        self._transforms.append(transform)
        self._link_to_id[link] = link_id
        self._parent[link_id] = self._link_to_id[parent]

    def get_mass_matrix(self) -> NDArray:
        nl = self.get_num_links()
        M = np.zeros((nl*6, nl*6))
        for i in range(nl):
            M[i*6:(i+1)*6, i*6:(i+1)*6] = self._links[i].get_mass_matrix()
        return M

    def get_num_links(self) -> int:
        return len(self._links)

    def get_parent_id(self, link: Link) -> int:
        return self._parent[self._link_to_id[link]]

    def link_to_id(self, link: Link) -> int:
        return self._link_to_id[link]
