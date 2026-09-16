"""A body with a non-spherical shape given by a formula: the aspherical
meshes, in MFEM's own format.

The body is a unit ball (disc in 2D) whose surface, and every sphere
inside it, is pushed along the radius by

    h(r, theta, phi) = eps * r * [ P2(cos theta) + beta * sin^2(theta) * cos(2 phi) ]

an oblate part and an elliptical equatorial part. In 2D the mesh lies in
the plane theta = pi/2, so only the elliptical part is seen. gmsh meshes
the sphere; the shape is applied to the nodes when the mesh is written
for MFEM, and the export checks that no element folds.

With --buffer a shell is appended outside the body, out to radius 1.2,
and the displacement is tapered to zero across it so that the outer
boundary stays a sphere, as the DtN and multipole conditions need. The
taper is quadratic in the shell and starts at the surface, which the
displacement declares as a knot: the mapping is continuous there but
kinked, and a kink is allowed only on a boundary the mesh honours.

Domain attribute 1 is the body (and 2 the buffer); boundary attribute 1
is the body's surface (and 2 the outer sphere). Order-2 elements, coarse
enough to build in seconds. Files: aspherical_{2d,3d}.mesh and
aspherical_buffer_{2d,3d}.mesh; `--all` builds the four.

Try them with the existing examples, e.g. from a build's examples/:
    ./anisotropic_elasticity -m ../data/aspherical_3d.mesh -o 1
    ./poisson_dtn -m ../data/aspherical_buffer_2d.mesh -o 2 -mth 1
    ./self_gravitating_elasticity -m ../data/aspherical_buffer_2d.mesh -o 2 -s 2
(love_numbers is not among them: it reads harmonics off the body's surface,
which must be a sphere, and says so.)
"""
import numpy as np

from planetmodel import CallableDisplacement, Geometry, Skeleton
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh, export_mfem_mesh)

from common import parser, report

EPS = 0.05     # amplitude of the shape, relative to the radius
BETA = 0.5     # elliptical part relative to the oblate part
BUFFER = 0.2   # thickness of the buffer shell relative to the body's radius

SIZING = {2: UniformInterfaces(0.1, 0.2, 1.0),
          3: UniformInterfaces(0.25, 0.5, 2.5)}


def shape(r, theta, phi):
    """The radial displacement of the body, growing with the radius."""
    p2 = 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
    return EPS * r * (p2 + BETA * np.sin(theta) ** 2 * np.cos(2.0 * phi))


def tapered_shape(r, theta, phi):
    """The same shape inside the body, tapered to zero across the buffer."""
    taper = np.clip((1.0 + BUFFER - r) / BUFFER, 0.0, 1.0) ** 2
    return shape(np.minimum(r, 1.0), theta, phi) * taper


def build(dim: int, buffer: bool, args) -> None:
    body = Geometry(Skeleton([0.0, 1.0]),
                    layer_names=["body"], interface_names=["surface"])
    if buffer:
        geometry = body.stretched(
            CallableDisplacement(tapered_shape, knots=[1.0], name="tapered shape"))
        spec = MeshSpec(geometry, SIZING[dim], dimension=dim, order=2,
                        shells=[Shell(ratio=BUFFER, name="buffer")],
                        outer_boundary="spherical")
        name = f"aspherical_buffer_{dim}d"
    else:
        geometry = body.stretched(CallableDisplacement(shape, name="shape"))
        spec = MeshSpec(geometry, SIZING[dim], dimension=dim, order=2)
        name = f"aspherical_{dim}d"

    # The reference (spherical) mesh, then the MFEM file with the nodes moved.
    reference = build_layered_mesh(spec, args.out / f"{name}_reference",
                                   verbose=args.verbose)
    exported = export_mfem_mesh(reference, args.out / name, delivery="physical")
    report(reference)
    print(f"{exported.mesh_path.name}: nodes moved by the shape; "
          f"smallest Jacobian ratio {exported.quality.get('min_ratio', float('nan')):.3f}")
    print(f"  manifest: {exported.manifest_path}")
    for p in (reference.msh_path, reference.manifest_path):
        p.unlink()


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--dim", type=int, default=2, choices=(2, 3),
                   help="2 for a disc, 3 for a ball")
    p.add_argument("--buffer", action="store_true",
                   help="add a buffer shell, with the shape tapered off across it")
    p.add_argument("--all", action="store_true",
                   help="build all four meshes")
    args = p.parse_args()

    if args.all:
        for dim in (2, 3):
            for buffer in (False, True):
                build(dim, buffer, args)
    else:
        build(args.dim, args.buffer, args)


if __name__ == "__main__":
    main()
