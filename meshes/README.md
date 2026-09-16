# Meshes

The gmsh meshes the examples and tests read are made here, with
[planetmodel](https://github.com/da380/planetmodel) driving gmsh, and
land in the build tree's `data/` directory; none is kept in the
repository. Each script builds one kind of mesh and is short enough to
read in a minute; copy the nearest one and change the skeleton, the
sizing or the shells to make a mesh of your own.

## In the build

`CMakeLists.txt` here runs the scripts as part of the build when
`GENERATE_MESHES` is on (the default whenever examples or tests are
built). One custom command per script writes its `.msh` and `.json`
files into `<build>/data`, so a mesh is generated once and again only
when its script or `common.py` changes. A new mesh is one `meshes_add`
line there.

The Python used is `MESHES_PYTHON` if set, else the one CMake finds; if
that cannot import `planetmodel.mesh3d`, a virtual environment is made
at `<build>/meshes-venv` and `planetmodel[meshing]` installed into it.
To reuse an environment you already have:

```
cmake -S . -B build -DMESHES_PYTHON=/path/to/python ...
```

## By hand

```
poetry install                       # planetmodel[meshing]: numpy, scipy, gmsh
poetry run python unit_disc.py -h    # any one script, with its options
poetry run python make_all.py --out /some/dir
```

Python 3.12 or later. Nothing here needs MFEM or a compiler; gmsh comes
as a wheel from PyPI. A script writes to the current directory unless
given `--out`, and never to the repository's `data/` (which is
gitignored for `.msh` and `.json` anyway).

## The scripts

| script | writes | consumers |
|---|---|---|
| `unit_disc.py` | `disk.msh` | transformed_diffusion |
| `unit_ball.py` | `ball.msh` | transformed_diffusion, anisotropic_elasticity |
| `offset_disc.py` | `circular_offset.msh` | poisson_dtn, submesh_injection |
| `disc_with_buffer.py` | `elastogravity_2d.msh` | self_gravitating_elasticity, love_numbers, tests |
| `ball_with_buffer.py` | `coupled_poisson.msh` | coupled_poisson, self_gravitating_elasticity, love_numbers, tests |
| `layered_earth.py --all` | `elastogravity_{two,three}_layer_2d.msh`, `elastogravity_three_layer_3d.msh` | elastogravity_layered, self_gravitating_relaxation, tests |
| `aspherical_body.py --all` | `aspherical_{2d,3d}.mesh`, `aspherical_buffer_{2d,3d}.mesh` | any example given `-m`; see the script |
| `make_all.py` | all of the above | |

`common.py` holds the two things every script shares: the default output
directory and the command line (`--out DIR`, `--verbose`).

## How a mesh is described

A **skeleton** is the list of boundary radii, from the centre outwards. A
**geometry** is a skeleton with names for its layers and interfaces. A
**shell** is a layer appended outside the geometry, used here as the buffer
region on whose outer boundary the far-field condition is applied. The
**sizing** gives every interface a target element size, a size far from
the interface and the distance over which one grows into the other. A
`MeshSpec` collects these with the dimension (2 for a disc, 3 for a ball)
and the element order, and `build_layered_mesh` does the rest: gmsh
geometry, physical groups, sizing fields, meshing, orientation, curving,
validation and the files on disk.

The offset disc is not spherically layered, so it has its own function,
`build_offset_mesh`, with the same sizing rules and the same output.

A **mapping** gives a geometry a non-spherical shape. `aspherical_body.py`
shows the pattern: a radial displacement `h(r, theta, phi)` from a
formula, wrapped in `CallableDisplacement` and attached with
`Geometry.stretched`. gmsh still meshes the sphere; `export_mfem_mesh`
moves the nodes when it writes the mesh in MFEM's own format, and refuses
a mapping that folds an element. With a buffer shell the displacement is
tapered to zero across it so that the outer boundary stays spherical for
the DtN and multipole conditions, and the surface, where the taper
starts, is declared a knot of the displacement. These meshes are
`.mesh` files and need the `mfem` extra (PyMFEM).

## What the files carry

Every gmsh mesh is MSH 2.2 with physical groups, which MFEM reads
directly; the aspherical ones are MFEM's native format with curved nodes.
Domain attributes number the layers 1..N from the centre, the buffer
last; boundary attributes number the interfaces 1..M in the same order,
so attribute 1 of a boundary element is the innermost interface and the
largest boundary attribute is the outer boundary of the domain. The
elements come out consistently oriented, so MFEM has nothing to fix on
load.

Beside each `.msh` file is a JSON **manifest** of the same name listing
the layers and interfaces with their attributes, names and radii. It is
what a reader consults instead of guessing what attribute 3 means; the
C++ side does not read it yet.

## Sizes and quality

Element sizes are in the mesh's own units, the unit radius here. The
sizes in the scripts were chosen to match the resolution of the meshes
they replaced. planetmodel checks every mesh before writing it and
refuses one with an inverted element; a warning is printed when the
worst element is usable but badly shaped. If a new sizing meets either,
nudging the size by a few per cent is usually enough.

Everything a script can do is documented in planetmodel; its tutorial
`04_a_mesh_for_mfem.py` walks through the same steps, including
topography through a mapping.
