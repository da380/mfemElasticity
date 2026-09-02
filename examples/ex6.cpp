
#include <cmath>
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
  auto mesh_1 = Mesh(mesh_file, 1, 1);
  auto dim = mesh_1.Dimension();
  {
    for (int l = 0; l < refinement; l++) {
      mesh_1.UniformRefinement();
    }
  }

  if (mesh_1.attributes.Max() < 2 || mesh_1.bdr_attributes.Max() < 2) {
    cerr << "\nInput mesh should have at least two materials and "
         << "two boundary attributes! (See schematic in ex2.cpp)\n"
         << endl;
    return 3;
  }

  // Set up the submesh.
  auto domain_0_marker = Array<int>(mesh_1.attributes.Max());
  domain_0_marker = 0;
  domain_0_marker[0] = 1;

  auto domain_0_bdr_marker = Array<int>(mesh_1.bdr_attributes.Max());
  domain_0_bdr_marker = 0;
  domain_0_bdr_marker[0] = 1;

  auto domain_1_bdr_marker = Array<int>(mesh_1.bdr_attributes.Max());
  domain_1_bdr_marker = 0;
  domain_1_bdr_marker[1] = 1;

  auto mesh_0 = SubMesh::CreateFromDomain(mesh_1, domain_0_marker);

  // Set up the finite element spaces.
  auto H1 = H1_FECollection(order, dim);
  auto fes_1 = FiniteElementSpace(&mesh_1, &H1);
  auto fes_0 = FiniteElementSpace(&mesh_0, &H1);

  // Get boundary DoFs on each mesh
  Array<int> boundary_0_dofs;
  fes_0.GetBoundaryTrueDofs(boundary_0_dofs);

  Array<int> boundary_1_dofs;
  fes_1.GetEssentialTrueDofs(domain_1_bdr_marker, boundary_1_dofs);

  // Set up the prolongation operator (fes_0 -> fes_1)
  auto P = SubMeshDofInjection(fes_0, fes_1).NewSparseMatrix();
  auto R = unique_ptr<SparseMatrix>(Transpose(*P));

  // Set up the bilinear forms.
  auto lambda = ConstantCoefficient(0.1);
  auto a_00 = BilinearForm(&fes_0);
  a_00.AddDomainIntegrator(new DiffusionIntegrator(lambda));
  a_00.Assemble();
  a_00.Finalize();

  auto a_11 = BilinearForm(&fes_1);
  a_11.AddDomainIntegrator(new DiffusionIntegrator());
  a_11.Assemble();
  a_11.Finalize();

  auto a_01 = BilinearForm(&fes_0);
  a_01.AddDomainIntegrator(new MassIntegrator());
  a_01.Assemble();
  a_01.Finalize();

  // 1. Get the raw blocks without applying BCs yet!
  auto &A_00 = a_00.SpMat();
  auto &A_11 = a_11.SpMat();
  auto &A_01_orig = a_01.SpMat();

  // 2. Perform custom matrix multiplication for off-diagonals
  std::unique_ptr<SparseMatrix> A_01(Mult(A_01_orig, *R));
  std::unique_ptr<SparseMatrix> A_10(Mult(*P, A_01_orig));

  // 3. Stitch them into a single monolithic SparseMatrix
  Array<int> block_offsets(3);
  block_offsets[0] = 0;
  block_offsets[1] = fes_0.GetVSize();
  block_offsets[2] = fes_0.GetVSize() + fes_1.GetVSize();

  BlockMatrix A_block(block_offsets);
  A_block.SetBlock(0, 0, &A_00);
  A_block.SetBlock(0, 1, A_01.get());
  A_block.SetBlock(1, 0, A_10.get());
  A_block.SetBlock(1, 1, &A_11);

  // Extract the single, massive SparseMatrix
  std::unique_ptr<SparseMatrix> A_global(A_block.CreateMonolithic());

  // 4. Set up the linear forms (Right-Hand Side)
  auto f_0 = FunctionCoefficient([](const Vector &x) { return 0; });
  auto b_0 = LinearForm(&fes_0);
  b_0.AddDomainIntegrator(new DomainLFIntegrator(f_0));
  b_0.Assemble();

  auto f_1 = FunctionCoefficient([](const Vector &x) {
    auto a0 = -0.75;
    auto a1 = -0.75;
    auto r2 = sqrt((x(0) - a0) * (x(0) - a0) + (x(1) - a1) * (x(1) - a1));
    return exp(-10 * r2);
  });
  auto b_1 = LinearForm(&fes_1);
  b_1.AddDomainIntegrator(new DomainLFIntegrator(f_1));
  b_1.Assemble();

  // 5. Build global RHS (B) and Solution (X) vectors
  int size_0 = fes_0.GetVSize();
  int size_1 = fes_1.GetVSize();
  int global_size = size_0 + size_1;

  Vector B_global(global_size);
  Vector X_global(global_size);
  B_global = 0.0;
  X_global = 0.0;  // This acts as our homogeneous (0.0) Dirichlet BC guess

  // Copy local RHS vectors into the global RHS vector using Elem() to bypass
  // the hidden operator
  for (int i = 0; i < size_0; i++) B_global(i) = b_0.Elem(i);
  for (int i = 0; i < size_1; i++) B_global(i + size_0) = b_1.Elem(i);

  // 6. Map your local boundary DoFs to global boundary DoFs
  Array<int> global_bdr_dofs;
  global_bdr_dofs.Append(boundary_0_dofs);
  for (int i = 0; i < boundary_1_dofs.Size(); i++) {
    global_bdr_dofs.Append(boundary_1_dofs[i] +
                           size_0);  // Shift by size of block 0
  }

  // 7. Apply Dirichlet BCs to the monolithic system
  // We loop through and eliminate them row by row from the matrix and RHS
  // vector
  for (int i = 0; i < global_bdr_dofs.Size(); i++) {
    int dof = global_bdr_dofs[i];
    // Safeguard: TrueDoF arrays can sometimes contain negative markers. We need
    // the positive index.
    if (dof < 0) dof = -1 - dof;

    A_global->EliminateRowCol(dof, X_global(dof), B_global);
  }

  // 8. Solve the global system using CG
  GSSmoother prec(*A_global);
  CGSolver solver;
  solver.SetRelTol(1e-8);
  solver.SetMaxIter(2000);
  solver.SetPrintLevel(1);
  solver.SetOperator(*A_global);
  solver.SetPreconditioner(prec);
  solver.Mult(B_global, X_global);

  // 9. Extract the solution back into GridFunctions for visualization
  auto u_0 = GridFunction(&fes_0);
  auto u_1 = GridFunction(&fes_1);

  for (int i = 0; i < size_0; i++) u_0(i) = X_global(i);
  for (int i = 0; i < size_1; i++) u_1(i) = X_global(i + size_0);

  // Visualise the results.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock.precision(8);
  sol_sock << "solution\n" << mesh_0 << u_0 << flush;
  sol_sock << "keys Rjlbc\n" << flush;

  socketstream sub_sol_sock(vishost, visport);
  sub_sol_sock.precision(8);
  sub_sol_sock << "solution\n" << mesh_1 << u_1 << flush;
  sub_sol_sock << "keys Rjlbc\n" << flush;
}
