#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

int main(int argc, char *argv[]) {
  // Initialise MPI and Hypre.
  Mpi::Init();
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  // Set default options.
  int dim = 2;
  int order = 1;
  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Deal with options.
  OptionsParser args(argc, argv);
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&serial_refinement, "-sr", "--serial_refinement",
                 "number of serial mesh refinements");
  args.AddOption(&parallel_refinement, "-pr", "--parallel_refinement",
                 "number of parallel mesh refinements");
  args.AddOption(&dim, "-dim", "--dimension", "dimension of the mesh");

  assert(dim == 2 || dim == 3);

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(cout);
  }

  // Read in mesh in serial.
  auto mesh_file =
      dim == 2
          ? string("/home/david/dev/meshing/examples/circular_offset.msh")
          : string("/home/david/dev/meshing/examples/spherical_offset.msh");
  auto mesh = Mesh(mesh_file, 1, 1);
  {
    for (int l = 0; l < serial_refinement; l++) {
      mesh.UniformRefinement();
    }
  }

  // Check the mesh has two attributes.
  assert(mesh.attributes.Max() == 2);

  // Form the parallel mesh.
  auto pmesh = ParMesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  {
    for (int l = 0; l < parallel_refinement; l++) {
      pmesh.UniformRefinement();
    }
  }

  auto x0 = MeshCentroid(&pmesh);

  auto [found, same, radius] = BoundaryRadius(&pmesh);
  if (Mpi::WorldRank() == 0) {
    x0.Print(cout);
    if (found == 1 && same == 1) {
      cout << radius << endl;
    } else {
      cout << "Boundary is not spherical about given point\n";
    }
  }
}