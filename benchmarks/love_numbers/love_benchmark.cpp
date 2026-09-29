// ============================================================================
// love_benchmark.cpp
//
// Load and tidal Love numbers of a spherically layered body, for comparison
// with a radial reference solution. The body is a case written by
// make_case.py (benchmark_case.hpp); nothing about the model is set here.
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
// At degree one the solution is known up to a rigid translation, which the
// solver fixes by leaving the displacement without rigid component in the
// true-dof inner product. The numbers and the coefficients written are
// those of the frame of the centre of mass of the body and its load, in
// which the potential outside has no part of degree one and k'_1 = -1: the
// translation d = phi_1 / g is added to U and V, and g(r) d taken from phi
// on the interface of radius r.
//
// The results are written as JSON, with the coefficients on every interface
// per unit forcing, the radial functions U, V and phi of each solution in
// every layer (the polynomial in the radius that, times the harmonic, is
// closest to the solution there), the sizes of the problem, the iteration
// counts and the times.
//
// Sample runs:
//    mpiexec -np 8 ./love_benchmark -c cases/homogeneous/case.json -lmax 6
//    mpiexec -np 8 ./love_benchmark -c case.json -o 3 -out results_o3.json
// ============================================================================

#include <fstream>

#include "benchmark_case.hpp"

using namespace mfem;
using namespace mfemElasticity;
using namespace benchmark;

namespace {

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

// The response to one forcing of one degree.
struct Response {
  bool converged = false;
  int outer = 0, inner = 0;
  double seconds = 0.0;
  real_t spurious = 0.0;
  // The translation to the centre-of-mass frame (degree one; else zero).
  real_t shift = 0.0;
  // (l, 0) coefficients of U, V and phi on each analysed interface.
  std::vector<real_t> u, v, phi;
  // The radial functions of the harmonic in each layer.
  std::vector<Case::LayerProfile> profiles;
};

void Write(std::ostream& os, const char* name, const Response& r,
           const std::string& extra) {
  os << "     \"" << name << "\": {" << extra
     << "\"converged\": " << (r.converged ? "true" : "false")
     << ", \"outer_iterations\": " << r.outer
     << ", \"inner_iterations\": " << r.inner
     << ", \"seconds\": " << Num(r.seconds)
     << ", \"spurious\": " << Num(r.spurious)
     << ", \"shift\": " << Num(r.shift) << ",\n      \"u\": " << List(r.u)
     << ",\n      \"v\": " << List(r.v) << ",\n      \"phi\": " << List(r.phi)
     << ",\n      \"profiles\": [";
  for (std::size_t j = 0; j < r.profiles.size(); j++) {
    const auto& p = r.profiles[j];
    os << (j ? "," : "") << "\n       {\"attribute\": " << p.attribute
       << ", \"radius\": " << List(p.radius) << ", \"u\": " << List(p.u)
       << ", \"v\": " << List(p.v) << ", \"phi\": " << List(p.phi) << "}";
  }
  os << "]}";
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();

  CaseOptions options;
  const char* out_file = "results.json";
  int lmin = 0;
  bool tide = true;
  bool profiles = true;

  OptionsParser args(argc, argv);
  options.Add(args);
  args.AddOption(&out_file, "-out", "--output", "Results file (JSON).");
  args.AddOption(&lmin, "-lmin", "--min-degree", "Lowest harmonic degree.");
  args.AddOption(&tide, "-tide", "--tide", "-no-tide", "--no-tide",
                 "Solve the tidal problems as well as the load problems.");
  args.AddOption(&profiles, "-profiles", "--profiles", "-no-profiles",
                 "--no-profiles",
                 "Write the radial functions of each solution by layer.");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  if (root) args.PrintOptions(std::cout);
  const int lmax = options.lmax;
  MFEM_VERIFY(lmin >= 0 && lmax >= lmin, "Degrees must satisfy 0 <= lmin <= "
                                         "lmax.");

  Case c(options);
  Problem& problem = *c.problem;
  const auto& basis = c.Basis();
  const real_t a = c.radius, g = c.gravity;

  // The gravity of the background at the radii of the profiles.
  std::vector<Vector> profile_gravity;
  if (profiles) {
    Vector none;
    ParGridFunction zero_u(c.fes_u.get()), zero_phi(c.fes_phi.get());
    zero_u = 0.0;
    zero_phi = 0.0;
    auto layout = c.Profiles(zero_u, zero_phi, 0);
    c.ProfileGravity(layout, profile_gravity);
  }

  auto solve = [&](int i, bool load) {
    Vector coefficients(basis.Size());
    coefficients = 0.0;
    coefficients[i] = 1.0;
    Response r;
    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = Clock::now();
    r.converged = c.Solve(coefficients, load);
    r.seconds = Seconds(t0);
    r.outer = problem.LastOuterIterations();
    r.inner = problem.LastInnerIterations();
    Vector cu, cv, cphi;
    for (const auto& an : c.analyses) {
      an.radial->Coefficients(problem.Displacement(), cu);
      an.tangential->Coefficients(problem.Displacement(), cv);
      an.scalar->Coefficients(problem.PotentialOnBody(), cphi);
      r.u.push_back(cu[i]);
      r.v.push_back(cv[i]);
      r.phi.push_back(cphi[i]);
      if (&an == &c.Surface()) {
        r.spurious = std::max(Spurious(cu, i), Spurious(cphi, i));
      }
    }
    if (profiles) {
      r.profiles = c.Profiles(problem.Displacement(), problem.Potential(), i);
    }
    if (basis.Degree(i) == 1) {
      r.shift = r.phi[c.surface] / g;
      for (std::size_t j = 0; j < r.profiles.size(); j++) {
        auto& p = r.profiles[j];
        for (int s = 0; s < p.radius.Size(); s++) {
          if (p.u.Size()) {
            p.u[s] += r.shift;
            p.v[s] += r.shift;
          }
          p.phi[s] -= profile_gravity[j][s] * r.shift;
        }
      }
      for (std::size_t j = 0; j < c.analyses.size(); j++) {
        r.u[j] += r.shift;
        r.v[j] += r.shift;
        r.phi[j] -= c.analyses[j].gravity * r.shift;
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
    c.Surface().scalar->Coefficients(phi_direct, cdirect);
    const real_t phi_sigma = cdirect[i];
    const real_t phi_exact = -4.0 * kPi * c.G * a / (2.0 * l + 1.0);
    const real_t h_load = -g * load.u[c.surface] / phi_sigma;
    const real_t k_load = load.phi[c.surface] / phi_sigma - 1.0;
    const real_t l_load = -g * load.v[c.surface] / phi_sigma;

    Response tidal;
    real_t h = NAN, k = NAN, l_tide = NAN;
    const bool with_tide = tide && l >= 2;
    if (with_tide) {
      tidal = solve(i, false);
      h = -g * tidal.u[c.surface];
      k = tidal.phi[c.surface];
      l_tide = -g * tidal.v[c.surface];
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
    os << "{\n";
    c.WriteHeader(os);
    os << ",\n  \"degrees\": [\n" << degrees.str() << "]\n}\n";
    std::cout << "\nWrote " << out_file << "\n";
  }
  return 0;
}
