# PyMUVs

**P**yMUVs **M**odels **U**nderwater **V**ehicles

A package for generating mathematical models of underwater vehicles. **PyMUVs**
is designed to deal with
- [x] Gravitational Forces
- [x] Buoyancy
- [x] Moment of Inertia
- [x] Added Mass
- [x] Hydrodynamic Damping (Linear and Quadratic)

The generated models are on the form

$M(q) \ddot q + C(q, \dot q) \dot q + D(q) \dot q + g(q) = \tau $

Where the matrices are available as [sympy](https://www.sympy.org/en/index.html) symbolic expressions.

## Installation

Clone the repository
```bash
git clone https://github.com/haakonbaa/pymuvs.git
```
and run
```bash
python3 -m pip install -e .
```
in the project root directory.

## Example

### Pendulum

```python
import sympy as sp
import numpy as np

from pymuvs import Link, Robot
from pymuvs.se3 import rot_x, trans

l = 1 # length of pendulum
m = 1 # mass of pendulum
g = 9.81 # gravity

theta, dtheta = sp.symbols('θ dθ')
Tbn = rot_x(theta) @ trans(0, 0, -l) # Transform from link- to inertial-frame
end_mass = Link(m, 0, np.zeros((3, 3)), np.zeros((6, 6)), np.zeros((6, 6)),
        np.zeros((6, 6)))

# Robot has links, each of which has their own transform
pendulum = Robot(links=[end_mass],
    transforms=[Tbn],
    params=[theta],
    diff_params=[dtheta])

model = pendulum.get_model(gvec=np.array([0, 0, -g])) # gravity in -z direction
print(f"{model.M=}")
print(f"{model.C=}")
print(f"{model.D=}")
print(f"{model.g=}")
```

The resulting output is
```txt
model.M = Matrix([[1.0]])
model.C = Matrix([[0]])
model.D = Matrix([[0]])
model.g = Matrix([[9.81*sin(θ)]])
```
which we recognize as the equations of motion for a pendulum:
$\ddot \theta + \frac{g}{l} \sin \theta = 0$
([wikipedia](https://en.wikipedia.org/wiki/Pendulum))

See the [examples](./examples) directory for more examples.

## Tests

Tests can be run by executing
```bash
python -m unittest discover -s test
```
in the project root directory.
