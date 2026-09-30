// ============================================================================
// field_benchmark.cpp
//
// The response of a spherically layered body to a load of many degrees and
// orders, compared with the reference solution as fields: the displacement
// over the solid and the potential perturbation over the body and the buffer
// shell. The body is a case written by make_case.py (benchmark_case.hpp) and
// the reference the radial solutions beside it (reference_field.hpp).
//
// The load is a smooth cap, sigma = cos^2(pi gamma / 2 alpha) within the
// angle alpha of its centre and nothing beyond, with gamma the angle from
// the centre; the centre is off the axes so that every order is excited.
// The cap is analysed into harmonic coefficients on the surface of the mesh
// and the load applied is their sum from degree lmin to lmax, so that the
// reference, which is the same coefficients times its radial functions, is
// the response to exactly the load the finite-element problem was given.
//
// The solution is known up to a rigid translation when the load has a part
// of degree one; it is taken to the frame of the centre of mass of the body
// and its load, that of the reference, by the translation that removes the
// degree-one part of the potential on the surface. With a fluid layer degree
// zero is not comparable (README.md) and lmin is one.
//
// Written: the relative L2 errors of u and phi, the coefficients of U, V and
// phi on the surface beside the reference's, as JSON; and with -pv the
// fields, the reference and their difference for ParaView.
//
// Sample runs:
//    mpiexec -np 8 ./field_benchmark -c case.json -o 3 -lmax 8
//    mpiexec -np 8 ./field_benchmark -c case.json -lat 60 -lon -40 -cap 20
//    -pv fields
// ============================================================================

#include <fstream>

#include "benchmark_case.hpp"
#include "reference_field.hpp"

using namespace mfem;
using namespace mfemElasticity;
using namespace benchmark;

namespace {

// The L2 norm of a coefficient over the mesh of a space.
real_t Norm(ParFiniteElementSpace& fes, Coefficient& f) {
  ParGridFunction zero(&fes);
  zero = 0.0;
  return zero.ComputeL2Error(f);
}

real_t Norm(ParFiniteElementSpace& fes, VectorCoefficient& f) {
  ParGridFunction zero(&fes);
  zero = 0.0;
  return zero.ComputeL2Error(f);
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const bool root = Mpi::Root();

  CaseOptions options;
  options.lmax = 8;
  const char* out_file = "field.json";
  const char* reference_file = "";
  const char* paraview = "";
  int lmin = -1;
  real_t latitude = 35.0, longitude = 25.0, cap = 30.0;

  OptionsParser args(argc, argv);
  options.Add(args);
  args.AddOption(&out_file, "-out", "--output", "Results file (JSON).");
  args.AddOption(&reference_file, "-ref", "--reference",
                 "Radial solutions of the reference (default: "
                 "reference_fields.txt beside the manifest).");
  args.AddOption(&lmin, "-lmin", "--min-degree",
                 "Lowest degree of the load (default: 0, or 1 with a fluid).");
  args.AddOption(&latitude, "-lat", "--latitude",
                 "Latitude of the cap's centre, in degrees.");
  args.AddOption(&longitude, "-lon", "--longitude",
                 "Longitude of the cap's centre, in degrees.");
  args.AddOption(&cap, "-cap", "--cap-radius",
                 "Angular radius of the cap, in degrees.");
  args.AddOption(&paraview, "-pv", "--paraview",
                 "Directory for ParaView output (none when empty).");
  args.Parse();
  if (!args.Good()) {
    if (root) args.PrintUsage(std::cout);
    return 1;
  }
  if (root) args.PrintOptions(std::cout);

  Case c(options);
  MFEM_VERIFY(c.eulerian,
              "field_benchmark supports the Eulerian methods (dahlen, "
              "gauged) for now.");
  Problem& problem = *c.problem;
  const auto& basis = c.Basis();
  const int n = basis.Size();
  if (lmin < 0) {
    lmin = c.HasFluid() ? 1 : 0;
  }
  MFEM_VERIFY(!(c.HasFluid() && lmin == 0),
              "Degree zero is not comparable with a fluid layer.");

  std::string reference_path = reference_file;
  if (reference_path.empty()) {
    const std::string manifest = options.manifest;
    const auto slash = manifest.find_last_of('/');
    reference_path =
        (slash == std::string::npos ? "" : manifest.substr(0, slash + 1)) +
        "reference_fields.txt";
  }
  const RadialReference reference(reference_path);

  // The cap and its coefficients on the surface of the mesh.
  const real_t deg = kPi / 180.0;
  Vector centre(3);
  centre[0] = std::cos(latitude * deg) * std::cos(longitude * deg);
  centre[1] = std::cos(latitude * deg) * std::sin(longitude * deg);
  centre[2] = std::sin(latitude * deg);
  const real_t alpha = cap * deg;
  FunctionCoefficient cap_c([&](const Vector& x) {
    const real_t cosine = (x * centre) / x.Norml2();
    const real_t gamma = std::acos(std::min(1.0, std::max(-1.0, cosine)));
    if (gamma >= alpha) {
      return 0.0;
    }
    const real_t s = std::cos(0.5 * kPi * gamma / alpha);
    return s * s;
  });
  Vector load;
  c.Surface().scalar->Coefficients(cap_c, load);
  for (int i = 0; i < n; i++) {
    if (basis.Degree(i) < lmin) {
      load[i] = 0.0;
    }
  }

  // The finite-element solution.
  MPI_Barrier(MPI_COMM_WORLD);
  const auto t0 = Clock::now();
  const bool converged = c.Solve(load, true);
  const double seconds = Seconds(t0);
  ParGridFunction u(static_cast<const ParGridFunction&>(
      problem.Displacement()));
  ParGridFunction phi(static_cast<const ParGridFunction&>(
      problem.Potential()));

  // To the frame of the centre of mass.
  Vector cu, cv, cphi;
  c.Surface().scalar->Coefficients(problem.PotentialOnBody(), cphi);
  const Vector shift = c.CentreOfMassShift(cphi);
  Vector translation = c.TranslationVector(shift);
  {
    VectorConstantCoefficient t_c(translation);
    ParGridFunction t_u(c.fes_u.get());
    t_u.ProjectCoefficient(t_c);
    u += t_u;
    TranslationPotential t_g(reference, translation);
    ParGridFunction t_phi(c.fes_phi.get());
    t_phi.ProjectCoefficient(t_g);
    phi += t_phi;
  }

  // The reference, and the errors.
  ReferenceDisplacement u_ref_c(reference, basis, load);
  ReferencePotential phi_ref_c(reference, basis, load);
  const real_t u_error = u.ComputeL2Error(u_ref_c);
  const real_t u_norm = Norm(*c.fes_u, u_ref_c);
  const real_t phi_error = phi.ComputeL2Error(phi_ref_c);
  const real_t phi_norm = Norm(*c.fes_phi, phi_ref_c);
  const real_t u_max_error = u.ComputeMaxError(u_ref_c);
  const real_t phi_max_error = phi.ComputeMaxError(phi_ref_c);

  // The coefficients on the surface, in the same frame.
  c.Surface().radial->Coefficients(problem.Displacement(), cu);
  c.Surface().tangential->Coefficients(problem.Displacement(), cv);
  cu += shift;
  cv += shift;
  cphi.Add(-c.gravity, shift);
  Vector ru(n), rv(n), rphi(n);
  const int top = reference.NumLayers();
  for (int i = 0; i < n; i++) {
    const int l = basis.Degree(i);
    ru[i] = load[i] * reference.Eval(top, RadialReference::U, l, c.radius);
    rv[i] = load[i] * reference.Eval(top, RadialReference::V, l, c.radius);
    rphi[i] =
        load[i] * reference.Eval(top, RadialReference::Phi, l, c.radius);
  }

  if (root) {
    std::cout << std::setprecision(6) << "\nCap at latitude " << latitude
              << ", longitude " << longitude << ", radius " << cap
              << " degrees; degrees " << lmin << " to " << options.lmax
              << "\n"
              << (converged ? "Converged" : "NOT CONVERGED") << " in "
              << problem.LastOuterIterations() << " iterations, " << seconds
              << " s\nTranslation to the centre-of-mass frame: "
              << translation[0] << " " << translation[1] << " "
              << translation[2] << "\n"
              << "Relative L2 error: u " << u_error / u_norm << ", phi "
              << phi_error / phi_norm << "\n";
  }

  if (paraview[0] != '\0') {
    ParGridFunction u_ref(c.fes_u.get()), u_diff(u);
    u_ref.ProjectCoefficient(u_ref_c);
    u_diff -= u_ref;
    ParaViewDataCollection solid_dc("solid", c.solid.get());
    solid_dc.SetPrefixPath(paraview);
    solid_dc.SetLevelsOfDetail(options.order);
    solid_dc.SetHighOrderOutput(true);
    solid_dc.RegisterField("u", &u);
    solid_dc.RegisterField("u_reference", &u_ref);
    solid_dc.RegisterField("u_difference", &u_diff);
    solid_dc.Save();

    ParGridFunction phi_ref(c.fes_phi.get()), phi_diff(phi);
    phi_ref.ProjectCoefficient(phi_ref_c);
    phi_diff -= phi_ref;
    ParaViewDataCollection ball_dc("ball", c.parent.get());
    ball_dc.SetPrefixPath(paraview);
    ball_dc.SetLevelsOfDetail(options.order);
    ball_dc.SetHighOrderOutput(true);
    ball_dc.RegisterField("phi", &phi);
    ball_dc.RegisterField("phi_reference", &phi_ref);
    ball_dc.RegisterField("phi_difference", &phi_diff);
    ball_dc.RegisterField("density", c.rho.get());
    ball_dc.Save();
  }

  if (root) {
    std::vector<real_t> degree(n), order(n);
    for (int i = 0; i < n; i++) {
      degree[i] = basis.Degree(i);
      order[i] = basis.Order(i);
    }
    // The load at a few directions, for a reader to check its harmonics
    // against.
    std::ostringstream samples;
    Vector x(3), Y;
    for (int k = 0; k < 12; k++) {
      const real_t theta = (0.3 + 0.23 * k) * deg * 60.0;
      const real_t lambda = (0.7 + 1.9 * k) * deg * 60.0;
      x[0] = std::sin(theta) * std::cos(lambda);
      x[1] = std::sin(theta) * std::sin(lambda);
      x[2] = std::cos(theta);
      basis.Eval(x, Y);
      samples << (k ? ", " : "") << "\n    {\"x\": " << List(x)
              << ", \"load\": " << Num(load * Y) << "}";
    }

    std::ofstream os(out_file);
    MFEM_VERIFY(os.good(), "Cannot write " << out_file << ".");
    os << "{\n";
    c.WriteHeader(os);
    os << ",\n  \"lmin\": " << lmin << ",\n  \"lmax\": " << options.lmax
       << ",\n  \"cap\": {\"latitude\": " << Num(latitude)
       << ", \"longitude\": " << Num(longitude) << ", \"radius\": "
       << Num(cap) << "},\n  \"converged\": "
       << (converged ? "true" : "false") << ",\n  \"outer_iterations\": "
       << problem.LastOuterIterations() << ",\n  \"seconds\": "
       << Num(seconds) << ",\n  \"translation\": " << List(translation)
       << ",\n  \"u_error\": " << Num(u_error / u_norm)
       << ",\n  \"phi_error\": " << Num(phi_error / phi_norm)
       << ",\n  \"u_norm\": " << Num(u_norm) << ",\n  \"phi_norm\": "
       << Num(phi_norm) << ",\n  \"u_max_error\": " << Num(u_max_error)
       << ",\n  \"phi_max_error\": " << Num(phi_max_error)
       << ",\n  \"degree\": " << List(degree) << ",\n  \"order_m\": "
       << List(order) << ",\n  \"load\": " << List(load)
       << ",\n  \"u\": " << List(cu) << ",\n  \"v\": " << List(cv)
       << ",\n  \"phi\": " << List(cphi) << ",\n  \"u_reference\": "
       << List(ru) << ",\n  \"v_reference\": " << List(rv)
       << ",\n  \"phi_reference\": " << List(rphi) << ",\n  \"samples\": ["
       << samples.str() << "]\n}\n";
    std::cout << "Wrote " << out_file << "\n";
  }
  return 0;
}
