
#include <fstream>
#include <memory>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

int main(int argc, char *argv[]) {
  // Set default options.
  const char *mesh_file = "../data/circular_offset.msh";
  int order = 1;
  int refinement = 0;
  int degree = 8;
  int residual = 0;
  int method = 0;
  int linearised = 0;

  // Deal with options.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&refinement, "-r", "--refinement",
                 "number of  mesh refinements");
  args.AddOption(&degree, "-deg", "--degree", "Order for Fourier exapansion");
  args.AddOption(&residual, "-res", "--residual",
                 "Output the residual from reference solution");
  args.AddOption(&method, "-mth", "--method",
                 "Solution method: 0 = Neuman, 1 = DtN, 2 = multipole.");
  args.AddOption(&linearised, "-lin", "--linearised",
                 "Solve reference (0) or linearised (1) problem.");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // Read in mesh.
  auto mesh = Mesh(mesh_file, 1, 1);
  auto dim = mesh.Dimension();
  {
    for (int l = 0; l < refinement; l++) {
      mesh.UniformRefinement();
    }
  }

  // Properties of the first attribute.
  auto dom_marker = Array<int>(mesh.attributes.Max());
  dom_marker = 0;
  dom_marker[0] = 1;
  auto bdr_marker = Array<int>(mesh.bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[0] = 1;
  auto c1 = MeshCentroid(&mesh, dom_marker);
  auto [found1, same1, r1] = SphericalBoundaryRadius(&mesh, bdr_marker, c1);

  // Properties of the full mesh.
  auto c2 = MeshCentroid(&mesh);
  auto [found2, same2, r2] = SphericalBoundaryRadius(&mesh, c2);

  // Build the SubMesh for the interior domain.
  auto subMesh = SubMesh::CreateFromDomain(mesh, dom_marker);

  // Set up the FE spaces (Vector-valued, vdim = dim)
  auto H1 = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &H1, dim);
  auto sub_fes = FiniteElementSpace(&subMesh, &H1, dim);

  // Create a Gridfunction on the whole space.
  auto u = GridFunction(&fes);
  auto f = VectorFunctionCoefficient(dim, [](const Vector &x, Vector &v) {
    v(0) = x[0] * x[1];
    if (v.Size() > 1) v(1) = x[0] + x[1];
    if (v.Size() > 2) v(2) = x[2];
  });
  u.ProjectCoefficient(f);

  // Set up the prolongation mapping
  auto P = SubMeshDofInjection(sub_fes, fes).NewSparseMatrix();

  // Use its transpose to restrict the GridFunction to the submesh
  auto u_sub = GridFunction(&sub_fes);
  P->MultTranspose(u, u_sub);

  // Visualise the results.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock.precision(8);
  sol_sock << "solution\n" << mesh << u << flush;
  sol_sock << "keys Rjlbc\n" << flush;

  socketstream sub_sol_sock(vishost, visport);
  sub_sol_sock.precision(8);
  sub_sol_sock << "solution\n" << subMesh << u_sub << flush;
  sub_sol_sock << "keys Rjlbc\n" << flush;

  // Now test the prolongation mapping by acting it
  // on a LinearForm defined initially on the SubMesh.
  Vector vec_one(dim);
  vec_one = 1.0;
  auto one = VectorConstantCoefficient(vec_one);

  // Create and assemble a LinearForm on the submesh
  auto b_sub = LinearForm(&sub_fes);
  b_sub.AddDomainIntegrator(new VectorDomainLFIntegrator(one));
  b_sub.Assemble();

  // Create a LinearForm (or just a Vector) on the parent mesh
  auto b = LinearForm(&fes);
  b = 0.0;  // Initialize with zeros

  // Extend the submesh LinearForm to the parent mesh: b_parent = P * b_sub
  P->Mult(b_sub, b);

  // Programmatic check: The sums should be identical
  auto sub_result = b_sub(u_sub);
  auto result = b(u);

  cout << "SubMesh LinearForm result: " << sub_result << endl;
  cout << "Parent  LinearForm result: " << result << endl;
  cout << "Difference:             " << abs(sub_result - result) << endl;
}