import gmsh
import math
import sys
import numpy as np

gmsh.initialize()
gmsh.option.setNumber("Mesh.Nodes", 1)
gmsh.option.setNumber("Mesh.VolumeFaces", 1)

gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

gmsh.model.add("circular_offset")


a = 1
b = 1.75

x0 = 0.5
y0 = 0.5

x1 = 0.3
y1 = 0.0

lc = 0.1


def meshSizeCallback(dim, tag, x, y, z, lc):

    r0 = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    r1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    small = 0.01
    big = 0.1
    fac = 0.3

    d0 = np.abs(r0 - a)
    d1 = np.abs(r1 - b)

    size = big

    if d0 < fac * a:
        size = small + (big - small) * d0 / (fac * a)

    if d1 < fac * b:
        size = min(size, (b / a) * (small + (big - small) * d1 / (fac * b)))

    return size


gmsh.model.mesh.setSizeCallback(meshSizeCallback)


def circle(x, y, r, lc):

    p1 = gmsh.model.geo.addPoint(x, y, 0, lc)
    p2 = gmsh.model.geo.addPoint(x + r, y, 0, lc)
    p3 = gmsh.model.geo.addPoint(x, y + r, 0, lc)
    p4 = gmsh.model.geo.addPoint(x - r, y, 0, lc)
    p5 = gmsh.model.geo.addPoint(x, y - r, 0, lc)

    c1 = gmsh.model.geo.addCircleArc(p2, p1, p3)
    c2 = gmsh.model.geo.addCircleArc(p3, p1, p4)
    c3 = gmsh.model.geo.addCircleArc(p4, p1, p5)
    c4 = gmsh.model.geo.addCircleArc(p5, p1, p2)

    l = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])

    return l, [c1, c2, c3, c4]


l1, b1 = circle(x0, y0, a, lc)
l2, b2 = circle(x1, y1, b, lc)

v1 = gmsh.model.geo.addPlaneSurface([l1])
v2 = gmsh.model.geo.addPlaneSurface([l1, l2])

gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(2, [v1], 1)
gmsh.model.addPhysicalGroup(2, [v2], 2)
gmsh.model.addPhysicalGroup(1, b1, 1)
gmsh.model.addPhysicalGroup(1, b2, 2)


gmsh.option.setNumber("Mesh.ElementOrder", 3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
# msh.model.setVisibility([(2,1),(2,2)], 1, recursive=True)


gmsh.model.mesh.generate(3)
gmsh.write("../../data/circular_offset.msh")


# Launch the GUI to see the results:
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
