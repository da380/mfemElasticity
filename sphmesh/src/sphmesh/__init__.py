"""sphmesh: parameterised gmsh meshing of geometrically spherical planetary models.

Scaffold only -- see doc/meshing_design.md (section 6) in the parent
repository for the plan. Intended module map:

  recipe      MeshRecipe: everything the mesher needs beyond the model
              (dimension, honoured interfaces, buffer layers, sizing,
              element order, algorithm choices)
  geometry    concentric OCC build, fragment, radius-sorted tagging
  sizing      per-interface Distance/Threshold background sizing fields
  mapping     radial diffeomorphisms and post-mesh node perturbation
  surface     lon-lat surface grids (CRUST-1.0 and friends), pending a
              longer-term home in sphmod
  sidecar     the .msh companion file: attributes <-> layers, radii,
              units, provenance
  benchmarks  offset two-sphere / two-disc benchmark geometries

The spherically symmetric model layer (Skeleton, Field, Model, deck IO,
exact PREM) comes from sphmod and is not reimplemented here.
"""

__version__ = "0.1.0"
