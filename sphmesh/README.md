# sphmesh

Parameterised gmsh meshing of *geometrically spherical* planetary models —
models diffeomorphic to a union of concentric spherical annuli (discs/annuli
in 2D) — for use with the mfemElasticity solvers.

This is the Python replacement for the C++ gmsh programs in `../meshing/`,
which are left untouched until this package reaches parity and are then to be
removed. Design and plan: `../doc/meshing_design.md`.

## Setup

Requires a sibling checkout of [sphmod](https://github.com/da380/sphmod)
(`~/dev/sphmod` next to this repository), which provides the spherically
symmetric model layer (`Skeleton`, `Field`, `Model`, deck IO, exact PREM).

```sh
cd sphmesh
poetry install
poetry run pytest
```
