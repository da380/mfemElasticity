import gmsh
import math
import sys
import numpy as np

gmsh.initialize()

gmsh.model.add("spherical_offset")

gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


x0 = 0.0
y0 = 0.25
z0 = 0.0

a = 0.5
b = 1
lc = 0.1


def meshSizeCallback(dim, tag, x, y, z, lc):

    r0 = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    r1 = np.sqrt(x**2 + y**2 + z**2)

    small = 0.025
    big = 0.05
    fac = 0.2

    d0 = np.abs(r0 - a)
    d1 = np.abs(r1 - b)

    size = big

    if d0 < fac * a:
        size = small + (big - small) * d0 / (fac * a)

    if d1 < fac * b:
        size = 2 * big - big * d1 / (fac * b)

    return size


gmsh.model.mesh.setSizeCallback(meshSizeCallback)


def sphere(x, y, z, r, lc):

    p1 = gmsh.model.geo.addPoint(x, y, z, lc)
    p2 = gmsh.model.geo.addPoint(x + r, y, z, lc)
    p3 = gmsh.model.geo.addPoint(x, y + r, z, lc)
    p4 = gmsh.model.geo.addPoint(x, y, z + r, lc)
    p5 = gmsh.model.geo.addPoint(x - r, y, z, lc)
    p6 = gmsh.model.geo.addPoint(x, y - r, z, lc)
    p7 = gmsh.model.geo.addPoint(x, y, z - r, lc)

    c1 = gmsh.model.geo.addCircleArc(p2, p1, p7)
    c2 = gmsh.model.geo.addCircleArc(p7, p1, p5)
    c3 = gmsh.model.geo.addCircleArc(p5, p1, p4)
    c4 = gmsh.model.geo.addCircleArc(p4, p1, p2)
    c5 = gmsh.model.geo.addCircleArc(p2, p1, p3)
    c6 = gmsh.model.geo.addCircleArc(p3, p1, p5)
    c7 = gmsh.model.geo.addCircleArc(p5, p1, p6)
    c8 = gmsh.model.geo.addCircleArc(p6, p1, p2)
    c9 = gmsh.model.geo.addCircleArc(p7, p1, p3)
    c10 = gmsh.model.geo.addCircleArc(p3, p1, p4)
    c11 = gmsh.model.geo.addCircleArc(p4, p1, p6)
    c12 = gmsh.model.geo.addCircleArc(p6, p1, p7)

    l1 = gmsh.model.geo.addCurveLoop([c5, c10, c4])
    l2 = gmsh.model.geo.addCurveLoop([c9, -c5, c1])
    l3 = gmsh.model.geo.addCurveLoop([c12, -c8, -c1])
    l4 = gmsh.model.geo.addCurveLoop([c8, -c4, c11])
    l5 = gmsh.model.geo.addCurveLoop([-c10, c6, c3])
    l6 = gmsh.model.geo.addCurveLoop([-c11, -c3, c7])
    l7 = gmsh.model.geo.addCurveLoop([-c2, -c7, -c12])
    l8 = gmsh.model.geo.addCurveLoop([-c6, -c9, c2])

    s1 = gmsh.model.geo.addSurfaceFilling([l1])
    s2 = gmsh.model.geo.addSurfaceFilling([l2])
    s3 = gmsh.model.geo.addSurfaceFilling([l3])
    s4 = gmsh.model.geo.addSurfaceFilling([l4])
    s5 = gmsh.model.geo.addSurfaceFilling([l5])
    s6 = gmsh.model.geo.addSurfaceFilling([l6])
    s7 = gmsh.model.geo.addSurfaceFilling([l7])
    s8 = gmsh.model.geo.addSurfaceFilling([l8])

    sl = gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4, s5, s6, s7, s8])

    return sl, [s1, s2, s3, s4, s5, s6, s7, s8]


sl1, l1 = sphere(x0, y0, z0, a, lc)
sl2, l2 = sphere(0, 0, 0, b, lc)


v1 = gmsh.model.geo.addVolume([sl1])
v2 = gmsh.model.geo.addVolume([sl2, sl1])

gmsh.model.occ.removeAllDuplicates()
gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(3, [v1], 1)
gmsh.model.addPhysicalGroup(3, [v2], 2)
gmsh.model.addPhysicalGroup(2, l1, 1)
gmsh.model.addPhysicalGroup(2, l2, 2)

gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)

gmsh.model.mesh.generate(3)
gmsh.write("../../data/spherical_offset.msh")


# Launch the GUI to see the results:
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
