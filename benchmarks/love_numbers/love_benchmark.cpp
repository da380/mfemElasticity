// ============================================================================
// love_benchmark.cpp
//
// Load and tidal Love numbers of a spherically layered body, for comparison
// with a radial reference solution. The body is a case written by
// make_case.py: a mesh of the body and its buffer shell, the density and the
// bulk and shear moduli as GridFunctions on it, and the manifest that says
// which layers are solid and which fluid, where the interfaces are and what
// G is in the model's units. Nothing about the model is set here.
//
// For each degree l (order 0) the surface load is set to sigma = Y_l0 and,
// separately for l >= 2, the tidal potential to psi = (r/a)^l Y_l0, with a
// the radius of the surface. The displacement, u = U Y r^ + V grad_1 Y, and
// the potential perturbation are analysed into harmonic coefficients on the
// surface and on every other interface that bounds a solid layer. With u_l,
// v_l and phi_l the (l, 0) coefficients of U, V and phi on the surface, g the
// surface gravity and phi_sigma the load's own potential on the surface
// (solved on the same mesh for the load alone, so that its discretisation
// error cancels in k'),
//
//   load:   h'_l = -g u_l / phi_sigma        k'_l = phi_l / phi_sigma - 1
//           l'_l = -g v_l / phi_sigma
//   tidal:  h_l  = -g u_l     l_l = -g v_l   k_l  = phi_l
//
// The displacement is in the solver's default gauge, without rigid component
// in the true-dof inner product. At degree one h', l' and k' depend on the
// frame, all by the same constant, and h' - k' and l' - k' are what compare.
//
// The results are written as JSON, with the coefficients on every interface
// per unit forcing, the sizes of the problem, the iteration counts and the
// times.
//
// Sample runs:
//    mpiexec -np 8 ./love_benchmark -c cases/homogeneous/case.json -lmax 6
//    mpiexec -np 8 ./love_benchmark -c case.json -o 3 -out results_o3.json
// ============================================================================

#include <mpi.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "mfemElasticity.hpp"

using namespace mfem;
using namespace mfemElasticity;

namespace {

constexpr real_t kPi = std::numbers::pi_v<real_t>;

using Clock = std::chrono::steady_clock;
using BHC = BoundaryHarmonicCoefficients;
using Problem = LinearQuasiStaticSelfGravitatingProblem;

double Seconds(Clock::time_point since) {
  return std::chrono::duration<double>(Clock::now() - since).count();
}

// The largest |c_i| over i != main, relative to |c_main|.
real_t Spurious(const Vector& c, int main) {
  real_t m = 0.0;
  for (int i = 0; i < c.Size(); i++) {
    if (i != main) {
      m = std::max(m, std::abs(c[i]));
    }
  }
  return m / (std::abs(c[main]) + 1e-300);
}

// The mean over each interface of the density on its fluid side: entry
// b - 1 for the boundary attribute b, zero for a boundary that is not a
// fluid-solid interface. The density is sampled at the vertices of the fluid
// elements that lie on the interface, so for a radial model the mean is the
// value there. The interface terms of the weak form want the fluid's density
// on boundary elements of the solid, which hold no fluid element to ask.
Vector FluidSideDensity(const MeshManifest& manifest, ParMesh& mesh,
                        const ParGridFunction& rho, int size) {
  Vector sum(size), count(size);
  sum = 0.0;
  count = 0.0;
  const real_t tol = 1e-6 * manifest.OuterRadius();
  Array<int> faces, orientations, face_vertices, element_vertices;
  for (const int fluid : manifest.FluidAttributes()) {
    for (const int b : manifest.FluidSolidInterfaces(fluid)) {
      const real_t radius =
          manifest.Interfaces()[b - manifest.Interfaces().front().attribute]
              .radius;
      for (int e = 0; e < mesh.GetNE(); e++) {
        if (mesh.GetAttribute(e) != fluid) {
          continue;
        }
        mesh.GetElementVertices(e, element_vertices);
        const IntegrationRule& corners =
            *Geometries.GetVertices(mesh.GetElementBaseGeometry(e));
        mesh.GetElementFaces(e, faces, orientations);
        for (const int face : faces) {
          mesh.GetFaceVertices(face, face_vertices);
          bool on_interface = true;
          for (const int v : face_vertices) {
            const real_t* x = mesh.GetVertex(v);
            const real_t r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
            on_interface = on_interface && std::abs(r - radius) < tol;
          }
          if (!on_interface) {
            continue;
          }
          for (const int v : face_vertices) {
            const int local = element_vertices.Find(v);
            sum[b - 1] += rho.GetValue(e, corners.IntPoint(local));
            count[b - 1] += 1.0;
          }
        }
      }
    }
  }
  Vector global_sum(size), global_count(size);
  MPI_Allreduce(sum.GetData(), global_sum.GetData(), size,
                MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
  MPI_Allreduce(count.GetData(), global_count.GetData(), size,
                MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
  for (int i = 0; i < size; i++) {
    global_sum[i] = global_count[i] > 0.0 ? global_sum[i] / global_count[i] : 0.0;
  }
  return global_sum;
}

// A number as JSON: null when not finite.
std::string Num(real_t x) {
  if (!std::isfinite(x)) {
    return "null";
  }
  std::ostringstream os;
  os << std::setprecision(16) << x;
  return os.str();
}

std::string List(const std::vector<real_t>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); i++) {
    s += (i ? ", " : "") + Num(v[i]);
  }
  return s + "]";
}

// The analysis of u_r and phi on one interface of the solid.
struct InterfaceAnalysis {
  MeshManifest::Interface interface;
  std::unique_ptr<BHC> radial, tangential, scalar;
};

// The response to one forcing of one degree.
struct Response {
  bool converged = false;
  int outer = 0, inner = 0;
  double seconds = 0.0;
  real_t spurious = 0.0;
  // (l, 0) coefficients of U, V and phi on each analysed interface.
  std::vector<real_t> u, v, phi;
};

void Write(std::ostream& os, const char* name, const Response& r,
           const std::string& extra) {
  os << "     \"" << name << "\": {" << extra
     << "\"converged\": " << (r.converged ? "true" : "false")
     << ", \"outer_iterations\": " << r.outer
     << ", \"inner_iterations\": " << r.inner
     << ", \"seconds\": " << Num(r.seconds)
     << ", \"spurious\": " << Num(r.spurious) << ",\n      \"u\": "
     << List(r.u) << ",\n      \"v\": " << List(r.v) << ",\n      \"phi\": "
     << List(r.phi) << "}";
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();

  const char* case_file = "case.json";
  const char* out_file = "results.json";
  int order = 2;
  int dtn_degree = 16;
  int lmin = 0, lmax = 6;
  int solver = 1;
  real_t rel_tol = 1e-10;
  bool tide = true;
  bool diagnostics = false;
  bool no_fluid_gradient = false;

  OptionsParser args(argc, argv);
  args.AddOption(&case_file, "-c", "--case", "Manifest of the case.");
  args.AddOption(&out_file, "-out", "--output", "Results file (JSON).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree", "DtN expansion degree.");
  args.AddOption(&lmin, "-lmin", "--min-degree", "Lowest harmonic degree.");
  args.AddOption(&lmax, "-lmax", "--max-degree", "Highest harmonic degree.");
  args.AddOption(&solver, "-s", "--solver",
                 "0: Schur-complement CG, 1: block MINRES.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
  args.AddOption(&tide, "-tide", "--tide", "-no-tide", "--no-tide",
                 "Solve the tidal problems as well as the load problems.");
  args.AddOption(&diagnostics, "-diag", "--diagnostics", "-no-diag",
                 "--no-diagnostics",
                 "Print rigid-mode residuals and potential-block Ritz values.");
  args.AddOption(&no_fluid_gradient, "-no-fluid-gradient",
                 "--no-fluid-gradient", "-fluid-gradient", "--fluid-gradient",
                 "Set d rho / d Phi_0 to zero in the fluid, as it is for a "
                 "uniform layer, instead of computing it from the fields.");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  if (root) args.PrintOptions(std::cout);
  MFEM_VERIFY(lmin >= 0 && lmax >= lmin, "Degrees must satisfy 0 <= lmin <= "
                                         "lmax.");
  MFEM_VERIFY(dtn_degree >= lmax, "The DtN degree must reach lmax.");

  const auto t_setup = Clock::now();

  // The case: the mesh and the fields are read in serial on every rank and
  // handed to the parallel mesh by the partitioning.
  const MeshManifest manifest(case_file);
  if (root) manifest.Print(std::cout);
  const real_t G = manifest.G();

  std::unique_ptr<ParMesh> parent;
  std::unique_ptr<ParGridFunction> rho, kappa, mu;
  long long n_elements = 0;
  {
    Mesh serial = manifest.LoadMesh();
    n_elements = serial.GetNE();
    auto rho_s = manifest.LoadField(serial, "rho");
    auto kappa_s = manifest.LoadField(serial, "kappa");
    auto mu_s = manifest.LoadField(serial, "mu");
    std::unique_ptr<int[]> partitioning(
        serial.GeneratePartitioning(Mpi::WorldSize()));
    parent = std::make_unique<ParMesh>(MPI_COMM_WORLD, serial,
                                       partitioning.get());
    auto distribute = [&](const GridFunction& f) {
      return std::make_unique<ParGridFunction>(parent.get(), &f,
                                               partitioning.get());
    };
    rho = distribute(*rho_s);
    kappa = distribute(*kappa_s);
    mu = distribute(*mu_s);
  }
  const int dim = parent->Dimension();
  MFEM_VERIFY(dim == 3, "The benchmark is for balls.");

  // The solid regions and the material on them.
  const Array<int> solid_attributes = manifest.SolidAttributes();
  const Array<int> fluid_attributes = manifest.FluidAttributes();
  ParSubMesh solid(ParSubMesh::CreateFromDomain(*parent, solid_attributes));
  auto on_solid = [&](const ParGridFunction& f,
                      std::unique_ptr<ParFiniteElementSpace>& fes) {
    fes = std::make_unique<ParFiniteElementSpace>(&solid,
                                                  f.ParFESpace()->FEColl());
    auto g = std::make_unique<ParGridFunction>(fes.get());
    ParSubMesh::Transfer(f, *g);
    return g;
  };
  std::unique_ptr<ParFiniteElementSpace> fes_rho, fes_kappa, fes_mu;
  const auto rho_solid = on_solid(*rho, fes_rho);
  const auto kappa_solid = on_solid(*kappa, fes_kappa);
  const auto mu_solid = on_solid(*mu, fes_mu);
  GridFunctionCoefficient rho_c(rho_solid.get()), kappa_c(kappa_solid.get()),
      mu_c(mu_solid.get()), rho_parent_c(rho.get());

  // The mass of the body and the gravity on its surface.
  real_t mass = 0.0;
  {
    ParLinearForm m(rho->ParFESpace());
    m.AddDomainIntegrator(new DomainLFIntegrator(rho_parent_c));
    m.Assemble();
    real_t local = m.Sum();
    MPI_Allreduce(&local, &mass, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                  MPI_COMM_WORLD);
  }
  const real_t a = manifest.SurfaceRadius();
  const real_t g = G * mass / (a * a);

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&solid, &fec, dim), fes_phi(parent.get(), &fec);
  const HYPRE_BigInt n_u = fes_u.GlobalTrueVSize();
  const HYPRE_BigInt n_phi = fes_phi.GlobalTrueVSize();
  if (root) {
    std::cout << "Ranks " << Mpi::WorldSize() << ", elements " << n_elements
              << ", displacement unknowns " << n_u << ", potential unknowns "
              << n_phi << "\nG " << G << ", radius " << a << ", mass " << mass
              << ", surface gravity " << g << "\n";
  }

  // The fluid layers: the density on the parent's fluid elements, and on
  // the solid's boundary elements the density of the fluid beyond them.
  const int n_bdr = solid.bdr_attributes.Max();
  Vector fluid_side = FluidSideDensity(manifest, *parent, *rho, n_bdr);
  PWConstCoefficient rho_interface_c(fluid_side);
  ConstantCoefficient zero_gradient(0.0);
  std::vector<FluidRegion> fluids;
  for (const int attribute : fluid_attributes) {
    FluidRegion f;
    f.attributes = Array<int>({attribute});
    f.density = &rho_parent_c;
    f.interface_density = &rho_interface_c;
    f.interface_marker = MeshManifest::Marker(
        manifest.FluidSolidInterfaces(attribute), n_bdr);
    if (no_fluid_gradient) {
      f.density_gradient = &zero_gradient;
    }
    fluids.push_back(f);
  }

  IsotropicElasticRheology rheology(dim, kappa_c, mu_c);
  Problem problem(&fes_u, &fes_phi, rheology, rho_c, G, dtn_degree, nullptr,
                  fluids);
  // A solid layer with fluid all around it turns freely in a spherical
  // model.
  for (const int attribute : solid_attributes) {
    bool enclosed = true;
    for (const auto& f : manifest.Interfaces()) {
      if (f.below == attribute || f.above == attribute) {
        const int other = f.below == attribute ? f.above : f.below;
        enclosed = enclosed && fluid_attributes.Find(other) >= 0;
      }
    }
    if (enclosed) {
      problem.AddRegionRotations(Array<int>({attribute}));
      if (root) {
        std::cout << "Layer " << attribute
                  << " is enclosed by fluid: its rotations are projected.\n";
      }
    }
  }
  problem.SetSolverType(solver == 0 ? Problem::SolverType::SchurCG
                                    : Problem::SolverType::BlockMINRES);
  problem.SetRelTol(rel_tol);

  // Harmonic analysis on every interface bounding a solid layer, the
  // surface among them.
  auto is_solid = [&](int attribute) {
    return solid_attributes.Find(attribute) >= 0;
  };
  std::vector<InterfaceAnalysis> analyses;
  int i_surface = -1;
  for (const auto& f : manifest.Interfaces()) {
    if (!is_solid(f.below) && !is_solid(f.above)) {
      continue;
    }
    const Array<int> marker =
        MeshManifest::Marker(Array<int>({f.attribute}), n_bdr);
    InterfaceAnalysis an;
    an.interface = f;
    an.radial = std::make_unique<BHC>(fes_u, marker, lmax,
                                      BHC::Component::Radial);
    an.tangential = std::make_unique<BHC>(fes_u, marker, lmax,
                                          BHC::Component::Tangential);
    an.scalar = std::make_unique<BHC>(problem.PotentialSpaceOnBody(), marker,
                                      lmax, BHC::Component::Scalar);
    if (f.attribute == manifest.SurfaceAttribute()) {
      i_surface = static_cast<int>(analyses.size());
    }
    analyses.push_back(std::move(an));
  }
  MFEM_VERIFY(i_surface >= 0, "The surface of the body is not solid.");
  const BHC& surface_radial = *analyses[i_surface].radial;
  const BHC& surface_scalar = *analyses[i_surface].scalar;
  const auto& basis = surface_radial.Basis();
  const Array<int>& surface = surface_radial.Marker();

  // One load and one tidal potential, with coefficients switched per degree.
  Vector zero(basis.Size());
  zero = 0.0;
  auto sigma = surface_radial.Expansion(zero, false);
  auto psi = surface_scalar.Expansion(zero, true);
  problem.SetSurfaceLoad(*sigma, surface);
  problem.SetTidalPotential(*psi);

  if (diagnostics) {
    const auto residuals = problem.RigidModeResiduals();
    real_t ritz_hi = 0.0;
    const real_t ritz_lo = fluids.empty() ? 0.0
                                          : problem.PotentialBlockMinEigenvalue(
                                                40, &ritz_hi);
    if (root && !fluids.empty()) {
      std::cout << "Potential block Ritz values: " << ritz_lo << " .. "
                << ritz_hi << (ritz_lo > 0.0 ? "" : "  (INDEFINITE)") << "\n";
    }
    if (root) {
      std::cout << "Rigid-mode residuals:";
      for (const auto r : residuals) {
        std::cout << " " << r;
      }
      std::cout << "\n";
    }
  }
  const double setup_seconds = Seconds(t_setup);

  auto solve = [&](int i, bool load) {
    Vector c(basis.Size());
    c = 0.0;
    c[i] = 1.0;
    sigma->SetCoefficients(load ? c : zero);
    psi->SetCoefficients(load ? zero : c);
    Response r;
    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = Clock::now();
    problem.AssembleForce(0.0);
    r.converged = problem.Solve();
    r.seconds = Seconds(t0);
    r.outer = problem.LastOuterIterations();
    r.inner = problem.LastInnerIterations();
    Vector cu, cv, cphi;
    for (const auto& an : analyses) {
      an.radial->Coefficients(problem.Displacement(), cu);
      an.tangential->Coefficients(problem.Displacement(), cv);
      an.scalar->Coefficients(problem.PotentialOnBody(), cphi);
      r.u.push_back(cu[i]);
      r.v.push_back(cv[i]);
      r.phi.push_back(cphi[i]);
      if (&an == &analyses[i_surface]) {
        r.spurious = std::max(Spurious(cu, i), Spurious(cphi, i));
      }
    }
    return r;
  };

  std::ostringstream degrees;
  ParGridFunction phi_direct(
      static_cast<ParFiniteElementSpace*>(&problem.PotentialSpaceOnBody()));
  if (root) {
    std::cout << std::setprecision(6)
              << "\n  l          h'          l'          k'           h"
              << "           l           k    spurious   phi_s  its     time\n";
  }
  for (int l = lmin; l <= lmax; l++) {
    const int i = basis.Index(l, 0);
    const Response load = solve(i, true);
    problem.SolveLoadPotential(phi_direct);
    Vector cdirect;
    surface_scalar.Coefficients(phi_direct, cdirect);
    const real_t phi_sigma = cdirect[i];
    const real_t phi_exact = -4.0 * kPi * G * a / (2.0 * l + 1.0);
    const real_t h_load = -g * load.u[i_surface] / phi_sigma;
    const real_t k_load = load.phi[i_surface] / phi_sigma - 1.0;
    const real_t l_load = -g * load.v[i_surface] / phi_sigma;

    Response tidal;
    real_t h = NAN, k = NAN, l_tide = NAN;
    const bool with_tide = tide && l >= 2;
    if (with_tide) {
      tidal = solve(i, false);
      h = -g * tidal.u[i_surface];
      k = tidal.phi[i_surface];
      l_tide = -g * tidal.v[i_surface];
    }

    if (root) {
      std::cout << std::setw(3) << l << std::setw(12) << h_load
                << std::setw(12) << l_load << std::setw(12) << k_load
                << std::setw(12) << h << std::setw(12) << l_tide
                << std::setw(12) << k << std::setw(12) << std::setprecision(2)
                << std::max(load.spurious, tidal.spurious) << std::setw(8)
                << std::setprecision(4) << phi_sigma / phi_exact
                << std::setw(5) << load.outer << std::setw(9)
                << std::setprecision(3) << load.seconds + tidal.seconds
                << std::setprecision(6)
                << (load.converged && (!with_tide || tidal.converged)
                        ? ""
                        : "   NOT CONVERGED")
                << "\n";
      degrees << (l > lmin ? ",\n" : "") << "    {\"degree\": " << l << ",\n";
      Write(degrees, "load", load,
            "\"h\": " + Num(h_load) + ", \"l\": " + Num(l_load) +
                ", \"k\": " + Num(k_load) +
                ", \"phi_direct\": " + Num(phi_sigma) +
                ", \"phi_direct_exact\": " + Num(phi_exact) + ", ");
      if (with_tide) {
        degrees << ",\n";
        Write(degrees, "tide", tidal,
              "\"h\": " + Num(h) + ", \"l\": " + Num(l_tide) +
                  ", \"k\": " + Num(k) + ", ");
      }
      degrees << "}";
    }
  }

  if (root) {
    std::ofstream os(out_file);
    MFEM_VERIFY(os.good(), "Cannot write " << out_file << ".");
    os << "{\n  \"case\": \"" << case_file << "\",\n  \"ranks\": "
       << Mpi::WorldSize() << ",\n  \"order\": " << order
       << ",\n  \"dtn_degree\": " << dtn_degree << ",\n  \"rel_tol\": "
       << Num(rel_tol) << ",\n  \"solver\": \""
       << (solver == 0 ? "schur_cg" : "block_minres")
       << "\",\n  \"elements\": " << n_elements
       << ",\n  \"displacement_unknowns\": " << n_u
       << ",\n  \"potential_unknowns\": " << n_phi << ",\n  \"G\": " << Num(G)
       << ",\n  \"radius\": " << Num(a) << ",\n  \"mass\": " << Num(mass)
       << ",\n  \"surface_gravity\": " << Num(g)
       << ",\n  \"setup_seconds\": " << Num(setup_seconds)
       << ",\n  \"surface\": " << i_surface << ",\n  \"interfaces\": [";
    for (std::size_t j = 0; j < analyses.size(); j++) {
      const auto& f = analyses[j].interface;
      os << (j ? ", " : "") << "\n    {\"attribute\": " << f.attribute
         << ", \"name\": \"" << f.name << "\", \"radius\": " << Num(f.radius)
         << ", \"measured_radius\": " << Num(analyses[j].radial->Radius())
         << "}";
    }
    os << "],\n  \"degrees\": [\n" << degrees.str() << "]\n}\n";
    std::cout << "\nWrote " << out_file << "\n";
  }
  return 0;
}
