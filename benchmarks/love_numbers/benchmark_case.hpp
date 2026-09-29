// ============================================================================
// benchmark_case.hpp
//
// What the drivers of the Love-number benchmark share: a case read from its
// manifest and set up as a LinearQuasiStaticSelfGravitatingProblem, the
// harmonic analysis of its solution on the interfaces of the solid, the
// translation that takes a solution to the centre-of-mass frame, and the
// writing of numbers as JSON.
//
// A case is what make_case.py writes: the mesh of the body and its buffer
// shell, the density and the bulk and shear moduli as L2 GridFunctions on
// it, and the manifest saying which layers are solid and which fluid, where
// the interfaces are and what G is in the model's units.
// ============================================================================

#pragma once

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

namespace benchmark {

using namespace mfem;
using namespace mfemElasticity;

inline constexpr real_t kPi = std::numbers::pi_v<real_t>;

using Clock = std::chrono::steady_clock;
using BHC = BoundaryHarmonicCoefficients;
using Problem = LinearQuasiStaticSelfGravitatingProblem;

// The fields of the model a case holds.
inline constexpr const char* kFields[] = {"rho", "kappa", "mu"};

// Where the parts of a case for a number of ranks are (partition_case.cpp).
inline std::string PartsDirectory(const MeshManifest& manifest, int ranks) {
  const std::string& path = manifest.Path();
  const auto slash = path.find_last_of('/');
  return (slash == std::string::npos ? "" : path.substr(0, slash + 1)) +
         "parts_" + std::to_string(ranks);
}

inline double Seconds(Clock::time_point since) {
  return std::chrono::duration<double>(Clock::now() - since).count();
}

// A number as JSON: null when not finite.
inline std::string Num(real_t x) {
  if (!std::isfinite(x)) {
    return "null";
  }
  std::ostringstream os;
  os << std::setprecision(16) << x;
  return os.str();
}

template <class V>
std::string List(const V& v) {
  std::string s = "[";
  for (int i = 0; i < static_cast<int>(v.size()); i++) {
    s += (i ? ", " : "") + Num(v[i]);
  }
  return s + "]";
}

inline std::string List(const Vector& v) {
  return List(std::vector<real_t>(v.begin(), v.end()));
}

// The density on the fluid side of each interface, from the manifest's
// one-sided values: entry b - 1 for the boundary attribute b, zero for a
// boundary that is not a fluid-solid interface. The interface terms of the
// weak form want the fluid's density on boundary elements of the solid,
// which hold no fluid element to ask; the manifest holds the model's exact
// value there.
inline Vector FluidSideDensity(const MeshManifest& manifest, int size) {
  Vector rho(size);
  rho = 0.0;
  const int first = manifest.Interfaces().front().attribute;
  for (const int fluid : manifest.FluidAttributes()) {
    for (const int b : manifest.FluidSolidInterfaces(fluid)) {
      rho[b - 1] = manifest.Interfaces()[b - first].ValueBeside("rho", fluid);
    }
  }
  return rho;
}

// The radial functions of one harmonic in one layer: for a field f and the
// harmonic Y_i, the polynomial a(r) of degree modes - 1 that is closest to
// the field in the layer,
//
//   scalar      minimising int (f - a Y_i)^2 dV
//   radial      minimising int (f . x^ - a Y_i)^2 dV
//   tangential  minimising int |f - (f . x^) x^ - a grad_1 Y_i|^2 dV
//
// over the elements of the given attribute. The polynomial is a sum of
// Chebyshev polynomials in the radius taken to [-1, 1] over the layer; what
// is returned are its values at the given radii. For a field that is a
// radial function times the harmonic this recovers the radial function.
enum class RadialPart { Scalar, Radial, Tangential };

inline Vector RadialFunction(ParMesh& mesh, int attribute, real_t r_inner,
                             real_t r_outer, const SurfaceHarmonics& basis,
                             int i, RadialPart part, Coefficient* scalar,
                             VectorCoefficient* vector, const Vector& radii,
                             int modes = 8, int quadrature_order = 12) {
  const int dim = mesh.Dimension();
  DenseMatrix A(modes);
  Vector b(modes), x(dim), Y, f(dim), T_k(modes);
  DenseMatrix gradY;
  A = 0.0;
  b = 0.0;
  auto chebyshev = [&](real_t r, Vector& T) {
    const real_t xi = std::min(
        1.0, std::max(-1.0, (2.0 * r - r_inner - r_outer) /
                                (r_outer - r_inner)));
    T[0] = 1.0;
    if (modes > 1) {
      T[1] = xi;
    }
    for (int k = 2; k < modes; k++) {
      T[k] = 2.0 * xi * T[k - 1] - T[k - 2];
    }
  };
  for (int e = 0; e < mesh.GetNE(); e++) {
    if (mesh.GetAttribute(e) != attribute) {
      continue;
    }
    ElementTransformation* T = mesh.GetElementTransformation(e);
    const IntegrationRule& ir =
        IntRules.Get(T->GetGeometryType(), quadrature_order);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      const real_t r = x.Norml2(), w = ip.weight * T->Weight();
      basis.EvalWithGradient(x, Y, gradY);
      chebyshev(r, T_k);
      real_t weight = 0.0, value = 0.0;
      if (part == RadialPart::Scalar) {
        weight = Y[i] * Y[i];
        value = scalar->Eval(*T, ip) * Y[i];
      } else {
        vector->Eval(f, *T, ip);
        if (part == RadialPart::Radial) {
          weight = Y[i] * Y[i];
          value = (f * x) / r * Y[i];
        } else {
          for (int c = 0; c < dim; c++) {
            weight += gradY(c, i) * gradY(c, i);
            value += f[c] * gradY(c, i);
          }
        }
      }
      for (int k = 0; k < modes; k++) {
        b[k] += w * value * T_k[k];
        for (int j = 0; j < modes; j++) {
          A(k, j) += w * weight * T_k[k] * T_k[j];
        }
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, A.GetData(), modes * modes,
                MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
  MPI_Allreduce(MPI_IN_PLACE, b.GetData(), modes,
                MPITypeMap<real_t>::mpi_type, MPI_SUM, mesh.GetComm());
  Vector values(radii.Size());
  values = 0.0;
  if (A.Trace() <= 0.0) {
    return values;
  }
  Vector a(modes);
  DenseMatrixInverse inverse(A);
  inverse.Mult(b, a);
  for (int s = 0; s < radii.Size(); s++) {
    chebyshev(radii[s], T_k);
    values[s] = a * T_k;
  }
  return values;
}

// The analysis of U, V and phi on one interface of the solid.
struct InterfaceAnalysis {
  MeshManifest::Interface interface;
  std::unique_ptr<BHC> radial, tangential, scalar;
  // The gravity of the background at the interface.
  real_t gravity = 0.0;
};

// What a driver asks of a case.
struct CaseOptions {
  const char* manifest = "case.json";
  int order = 2;
  int dtn_degree = 16;
  int lmax = 6;
  int solver = 1;
  real_t rel_tol = 1e-10;
  bool no_fluid_gradient = false;
  bool diagnostics = false;
  bool gauged = false;
  real_t gauge_eps = 1e-2;
  int gauge_refinements = 3;
  const char* cmb = "full";

  void Add(OptionsParser& args) {
    args.AddOption(&cmb, "-cmb", "--cmb-approximation",
                   "Fluid-interface treatment of the Dahlen path: 'full' "
                   "(stratified, F1+F2+F3), 'nomass' (drop the fluid mass "
                   "term F1), 'uniform' (nomass with a constant fluid-side "
                   "density, the region's outermost interface value: the "
                   "standard unmeshed-core condition of the GIA codes), "
                   "'winkler' (uniform without the interface potential "
                   "coupling F3: buoyancy alone).");
    args.AddOption(&gauged, "-gauged", "--gauged", "-dahlen", "--dahlen",
                   "Gauged fluid treatment: the fluid layers join the "
                   "displacement SubMesh with their bulk modulus and a gauge "
                   "shear penalty (doc/gauged_fluid.md), instead of Dahlen's "
                   "interface and fluid-mass terms.");
    args.AddOption(&gauge_eps, "-geps", "--gauge-epsilon",
                   "Gauge penalty factor epsilon (gauged fluid).");
    args.AddOption(&gauge_refinements, "-gref", "--gauge-refinements",
                   "Tikhonov refinement steps per solve (gauged fluid).");
    args.AddOption(&manifest, "-c", "--case", "Manifest of the case.");
    args.AddOption(&order, "-o", "--order", "Finite element order.");
    args.AddOption(&dtn_degree, "-deg", "--dtn-degree",
                   "DtN expansion degree.");
    args.AddOption(&lmax, "-lmax", "--max-degree", "Highest harmonic degree.");
    args.AddOption(&solver, "-s", "--solver",
                   "0: Schur-complement CG, 1: block MINRES.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
    args.AddOption(&no_fluid_gradient, "-no-fluid-gradient",
                   "--no-fluid-gradient", "-fluid-gradient",
                   "--fluid-gradient",
                   "Set d rho / d Phi_0 to zero in the fluid, as it is for a "
                   "uniform layer, instead of computing it from the fields.");
    args.AddOption(&diagnostics, "-diag", "--diagnostics", "-no-diag",
                   "--no-diagnostics",
                   "Print rigid-mode residuals and potential-block Ritz "
                   "values.");
  }
};

// A case set up for solving. When the case has been partitioned for the
// number of ranks each rank reads its own part; else the mesh and the fields
// are read in serial on every rank and handed to the parallel mesh by the
// partitioning.
class Case {
 public:
  explicit Case(const CaseOptions& options)
      : options_(options), manifest(options.manifest) {
    const bool root = Mpi::Root();
    const auto start = Clock::now();
    MFEM_VERIFY(options.dtn_degree >= options.lmax,
                "The DtN degree must reach lmax.");
    if (root) manifest.Print(std::cout);
    G = manifest.G();
    const int rank = Mpi::WorldRank();
    const std::string parts = PartsDirectory(manifest, Mpi::WorldSize());
    if (std::ifstream(MakeParFilename(parts + "/mesh.", rank)).good()) {
      if (root) {
        std::cout << "Reading the parts in " << parts << "\n";
      }
      std::ifstream mesh_in(MakeParFilename(parts + "/mesh.", rank));
      parent = std::make_unique<ParMesh>(MPI_COMM_WORLD, mesh_in, false, 1,
                                         false);
      elements = parent->GetGlobalNE();
      auto read = [&](const char* name) {
        std::ifstream in(
            MakeParFilename(parts + "/" + name + ".", rank));
        MFEM_VERIFY(in.good(), "A part of " << name << " is missing.");
        return std::make_unique<ParGridFunction>(parent.get(), in);
      };
      rho = read("rho");
      kappa = read("kappa");
      mu = read("mu");
    } else {
      Mesh serial = manifest.LoadMesh();
      elements = serial.GetNE();
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
    dim = parent->Dimension();
    MFEM_VERIFY(dim == 3, "The benchmark is for balls.");

    // The displacement regions and the material on them: the solid layers,
    // and with the gauged fluid treatment the fluid layers as well (the
    // member keeps the name `solid` as "the displacement SubMesh").
    solid_attributes = manifest.SolidAttributes();
    fluid_attributes = manifest.FluidAttributes();
    Array<int> u_attributes(solid_attributes);
    if (options.gauged) {
      u_attributes.Append(fluid_attributes);
      u_attributes.Sort();
    }
    solid = std::make_unique<ParSubMesh>(
        ParSubMesh::CreateFromDomain(*parent, u_attributes));
    rho_solid_ = OnSolid(*rho);
    kappa_solid_ = OnSolid(*kappa);
    mu_solid_ = OnSolid(*mu);
    rho_c_ = std::make_unique<GridFunctionCoefficient>(rho_solid_.get());
    kappa_c_ = std::make_unique<GridFunctionCoefficient>(kappa_solid_.get());
    mu_c_ = std::make_unique<GridFunctionCoefficient>(mu_solid_.get());
    rho_parent_c_ = std::make_unique<GridFunctionCoefficient>(rho.get());

    // The mass of the body and the gravity on its surface.
    {
      ParLinearForm m(rho->ParFESpace());
      m.AddDomainIntegrator(new DomainLFIntegrator(*rho_parent_c_));
      m.Assemble();
      real_t local = m.Sum();
      MPI_Allreduce(&local, &mass, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    MPI_COMM_WORLD);
    }
    radius = manifest.SurfaceRadius();
    gravity = G * mass / (radius * radius);

    fec_ = std::make_unique<H1_FECollection>(options.order, dim);
    fes_u = std::make_unique<ParFiniteElementSpace>(solid.get(), fec_.get(),
                                                    dim);
    fes_phi = std::make_unique<ParFiniteElementSpace>(parent.get(),
                                                      fec_.get());
    displacement_unknowns = fes_u->GlobalTrueVSize();
    potential_unknowns = fes_phi->GlobalTrueVSize();
    if (root) {
      std::cout << "Ranks " << Mpi::WorldSize() << ", elements " << elements
                << ", displacement unknowns " << displacement_unknowns
                << ", potential unknowns " << potential_unknowns << "\nG "
                << G << ", radius " << radius << ", mass " << mass
                << ", surface gravity " << gravity << "\n";
    }

    // The fluid layers: the density on the parent's fluid elements, and on
    // the solid's boundary elements the density of the fluid beyond them.
    const std::string cmb(options.cmb);
    MFEM_VERIFY(cmb == "full" || cmb == "nomass" || cmb == "uniform" ||
                    cmb == "winkler",
                "-cmb must be full, nomass, uniform or winkler.");
    MFEM_VERIFY(!options.gauged || cmb == "full",
                "-cmb applies to the Dahlen path only.");
    const int n_bdr = solid->bdr_attributes.Max();
    std::vector<FluidRegion> fluids;
    if (!options.gauged) {
      fluid_side_ = FluidSideDensity(manifest, n_bdr);
      rho_interface_c_ = std::make_unique<PWConstCoefficient>(fluid_side_);
      for (const int attribute : fluid_attributes) {
        FluidRegion f;
        f.attributes = Array<int>({attribute});
        f.density = rho_parent_c_.get();
        f.interface_density = rho_interface_c_.get();
        f.interface_marker = MeshManifest::Marker(
            manifest.FluidSolidInterfaces(attribute), n_bdr);
        if (options.no_fluid_gradient || cmb != "full") {
          f.density_gradient = &zero_;
        }
        if (cmb == "uniform" || cmb == "winkler") {
          // The constant fluid density of the approximate condition: the
          // fluid-side value at the region's outermost interface (the
          // core-top density for a core).
          const int first = manifest.Interfaces().front().attribute;
          real_t rho_c = 0.0, r_max = -1.0;
          for (const int b : manifest.FluidSolidInterfaces(attribute)) {
            const auto& itf = manifest.Interfaces()[b - first];
            if (itf.radius > r_max) {
              r_max = itf.radius;
              rho_c = itf.ValueBeside("rho", attribute);
            }
          }
          cmb_rho_.push_back(std::make_unique<ConstantCoefficient>(rho_c));
          f.interface_density = cmb_rho_.back().get();
          if (root) {
            std::cout << "CMB approximation '" << cmb << "': fluid layer "
                      << attribute << " with constant interface density "
                      << rho_c << "\n";
          }
        }
        f.interface_potential_coupling = cmb != "winkler";
        fluids.push_back(f);
      }
    }

    rheology_ = std::make_unique<IsotropicElasticRheology>(dim, *kappa_c_,
                                                           *mu_c_);
    problem = std::make_unique<Problem>(fes_u.get(), fes_phi.get(),
                                        *rheology_, *rho_c_, G,
                                        options.dtn_degree, nullptr, fluids);
    if (options.gauged && fluid_attributes.Size() > 0) {
      // The gauge shear scale is the model's own bulk modulus (the fluid's
      // kappa on the fluid layers).
      gauge_marker_.SetSize(solid->attributes.Max());
      gauge_marker_ = 0;
      for (const int a : fluid_attributes) {
        gauge_marker_[a - 1] = 1;
      }
      problem->SetGaugedFluid(gauge_marker_, *kappa_c_, options.gauge_eps,
                              options.gauge_refinements);
      if (root) {
        std::cout << "Gauged fluid: eps " << options.gauge_eps << ", "
                  << options.gauge_refinements << " refinements.\n";
      }
    }
    // A solid layer with fluid all around it turns freely in a spherical
    // model; with the gauged fluid the penalty owns those modes and they
    // must not be projected (doc/gauged_fluid.md).
    if (!options.gauged) {
      for (const int attribute : solid_attributes) {
        bool enclosed = true;
        for (const auto& f : manifest.Interfaces()) {
          if (f.below == attribute || f.above == attribute) {
            const int other = f.below == attribute ? f.above : f.below;
            enclosed = enclosed && fluid_attributes.Find(other) >= 0;
          }
        }
        if (enclosed) {
          problem->AddRegionRotations(Array<int>({attribute}));
          if (root) {
            std::cout << "Layer " << attribute
                      << " is enclosed by fluid: its rotations are "
                         "projected.\n";
          }
        }
      }
    }
    problem->SetSolverType(options.solver == 0
                               ? Problem::SolverType::SchurCG
                               : Problem::SolverType::BlockMINRES);
    problem->SetRelTol(options.rel_tol);

    // Harmonic analysis on every interface bounding a solid layer, the
    // surface among them.
    for (const auto& f : manifest.Interfaces()) {
      if (solid_attributes.Find(f.below) < 0 &&
          solid_attributes.Find(f.above) < 0) {
        continue;
      }
      const Array<int> marker =
          MeshManifest::Marker(Array<int>({f.attribute}), n_bdr);
      InterfaceAnalysis an;
      an.interface = f;
      an.radial = std::make_unique<BHC>(*fes_u, marker, options.lmax,
                                        BHC::Component::Radial);
      an.tangential = std::make_unique<BHC>(*fes_u, marker, options.lmax,
                                            BHC::Component::Tangential);
      an.scalar = std::make_unique<BHC>(problem->PotentialSpaceOnBody(),
                                        marker, options.lmax,
                                        BHC::Component::Scalar);
      if (f.attribute == manifest.SurfaceAttribute()) {
        surface = static_cast<int>(analyses.size());
      }
      analyses.push_back(std::move(an));
    }
    MFEM_VERIFY(surface >= 0, "The surface of the body is not solid.");
    // The gravity of the background at the radius of each interface,
    // G M(r) / r^2 with M(r) the mass of the layers within.
    {
      const int n_layers = parent->attributes.Max();
      Vector local(n_layers), layer_mass(n_layers);
      for (int attribute = 1; attribute <= n_layers; attribute++) {
        Array<int> marker(n_layers);
        marker = 0;
        marker[attribute - 1] = 1;
        ParLinearForm m(rho->ParFESpace());
        m.AddDomainIntegrator(new DomainLFIntegrator(*rho_parent_c_), marker);
        m.Assemble();
        local[attribute - 1] = m.Sum();
      }
      MPI_Allreduce(local.GetData(), layer_mass.GetData(), n_layers,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
      for (auto& an : analyses) {
        real_t within = 0.0;
        for (int attribute = 1; attribute <= an.interface.below; attribute++) {
          within += layer_mass[attribute - 1];
        }
        const real_t r = an.interface.radius;
        an.gravity = G * within / (r * r);
      }
    }

    // One load and one tidal potential, their coefficients set per solve.
    Vector zero(Basis().Size());
    zero = 0.0;
    sigma = analyses[surface].radial->Expansion(zero, false);
    psi = analyses[surface].scalar->Expansion(zero, true);
    problem->SetSurfaceLoad(*sigma, analyses[surface].radial->Marker());
    problem->SetTidalPotential(*psi);

    if (options.diagnostics) {
      const auto residuals = problem->RigidModeResiduals();
      real_t hi = 0.0;
      const real_t lo = fluids.empty()
                            ? 0.0
                            : problem->PotentialBlockMinEigenvalue(40, &hi);
      if (root) {
        if (!fluids.empty()) {
          std::cout << "Potential block Ritz values: " << lo << " .. " << hi
                    << (lo > 0.0 ? "" : "  (INDEFINITE)") << "\n";
        }
        std::cout << "Rigid-mode residuals:";
        for (const auto r : residuals) {
          std::cout << " " << r;
        }
        std::cout << "\n";
      }
    }
    setup_seconds = Seconds(start);
  }

  const SurfaceHarmonics& Basis() const {
    return analyses[surface].radial->Basis();
  }
  const InterfaceAnalysis& Surface() const { return analyses[surface]; }
  bool HasFluid() const { return fluid_attributes.Size() > 0; }

  // Solve for the surface load or the tidal potential of the given
  // coefficients; returns the solver's convergence.
  bool Solve(const Vector& coefficients, bool load) {
    Vector zero(coefficients.Size());
    zero = 0.0;
    sigma->SetCoefficients(load ? coefficients : zero);
    psi->SetCoefficients(load ? zero : coefficients);
    problem->AssembleForce(0.0);
    return problem->Solve();
  }

  // The rigid translation that takes the solution to the frame of the
  // centre of mass of the body and its load, in which the potential
  // perturbation outside has no part of degree one: the coefficients d of
  // d . x^ = sum_m d_m Y_1m, from the degree-one coefficients of phi on the
  // surface. A translation d adds d to U and to V at degree one and takes
  // g(r) d from phi, with g the gravity at that radius.
  Vector CentreOfMassShift(const Vector& phi_surface) const {
    const auto& basis = Basis();
    Vector d(basis.Size());
    d = 0.0;
    if (basis.MaxDegree() >= 1) {
      for (int m = -1; m <= 1; m++) {
        const int i = basis.Index(1, m);
        d[i] = phi_surface[i] / gravity;
      }
    }
    return d;
  }

  // The vector t with t . x^ = sum_m d_m Y_1m(x^).
  Vector TranslationVector(const Vector& d) const {
    const auto& basis = Basis();
    Vector t(dim), e(dim), Y;
    t = 0.0;
    if (basis.MaxDegree() < 1) {
      return t;
    }
    for (int j = 0; j < dim; j++) {
      e = 0.0;
      e[j] = 1.0;
      basis.Eval(e, Y);
      for (int m = -1; m <= 1; m++) {
        const int i = basis.Index(1, m);
        t[j] += d[i] * Y[i];
      }
    }
    return t;
  }

  // The radial functions U, V and phi of the harmonic i of a solution, in
  // every layer of the mesh, at `samples` radii of each: the displacement
  // in the solid layers, the potential in all of them and in the buffer.
  struct LayerProfile {
    int attribute = 0;
    Vector radius, u, v, phi;
  };

  std::vector<LayerProfile> Profiles(const GridFunction& u,
                                     const GridFunction& phi, int i,
                                     int samples = 17) {
    VectorGridFunctionCoefficient u_c(&u);
    GridFunctionCoefficient phi_c(&phi);
    const auto& basis = Basis();
    std::vector<LayerProfile> out;
    for (const auto& layer : manifest.Layers()) {
      LayerProfile p;
      p.attribute = layer.attribute;
      p.radius.SetSize(samples);
      for (int s = 0; s < samples; s++) {
        p.radius[s] = layer.r_inner +
                      (layer.r_outer - layer.r_inner) * s / (samples - 1.0);
      }
      p.phi = RadialFunction(*parent, layer.attribute, layer.r_inner,
                             layer.r_outer, basis, i, RadialPart::Scalar,
                             &phi_c, nullptr, p.radius);
      if (solid_attributes.Find(layer.attribute) >= 0) {
        p.u = RadialFunction(*solid, layer.attribute, layer.r_inner,
                             layer.r_outer, basis, i, RadialPart::Radial,
                             nullptr, &u_c, p.radius);
        if (basis.Degree(i) > 0) {
          p.v = RadialFunction(*solid, layer.attribute, layer.r_inner,
                               layer.r_outer, basis, i,
                               RadialPart::Tangential, nullptr, &u_c,
                               p.radius);
        } else {
          p.v.SetSize(samples);
          p.v = 0.0;
        }
      }
      out.push_back(std::move(p));
    }
    return out;
  }

  // The gravity of the background at the radii of the profiles, from the
  // radial derivative of the background potential.
  void ProfileGravity(std::vector<LayerProfile>& profiles,
                      std::vector<Vector>& gravity) {
    GradientGridFunctionCoefficient grad(&problem->BackgroundPotential());
    gravity.clear();
    for (const auto& p : profiles) {
      const auto& layer = manifest.Layers()[p.attribute - 1];
      Vector g = RadialFunction(*parent, layer.attribute, layer.r_inner,
                                layer.r_outer, Basis(), 0,
                                RadialPart::Radial, nullptr, &grad, p.radius);
      g *= 1.0 / std::sqrt(4.0 * kPi);
      gravity.push_back(g);
    }
  }

  // Common entries of a results file (no collective call: the root alone
  // writes).
  void WriteHeader(std::ostream& os) const {
    os << "  \"case\": \"" << options_.manifest << "\",\n  \"ranks\": "
       << Mpi::WorldSize() << ",\n  \"order\": " << options_.order
       << ",\n  \"dtn_degree\": " << options_.dtn_degree
       << ",\n  \"rel_tol\": " << Num(options_.rel_tol)
       << ",\n  \"solver\": \""
       << (options_.solver == 0 ? "schur_cg" : "block_minres")
       << "\",\n  \"fluid_treatment\": \""
       << (options_.gauged ? "gauged" : "dahlen") << "\""
       << ",\n  \"cmb\": \"" << options_.cmb << "\""
       << (options_.gauged
               ? ",\n  \"gauge_epsilon\": " + Num(options_.gauge_eps) +
                     ",\n  \"gauge_refinements\": " +
                     std::to_string(options_.gauge_refinements)
               : std::string())
       << ",\n  \"elements\": " << elements
       << ",\n  \"displacement_unknowns\": " << displacement_unknowns
       << ",\n  \"potential_unknowns\": " << potential_unknowns
       << ",\n  \"G\": " << Num(G) << ",\n  \"radius\": " << Num(radius)
       << ",\n  \"mass\": " << Num(mass) << ",\n  \"surface_gravity\": "
       << Num(gravity) << ",\n  \"setup_seconds\": " << Num(setup_seconds)
       << ",\n  \"surface\": " << surface << ",\n  \"interfaces\": [";
    for (std::size_t j = 0; j < analyses.size(); j++) {
      const auto& f = analyses[j].interface;
      os << (j ? ", " : "") << "\n    {\"attribute\": " << f.attribute
         << ", \"name\": \"" << f.name << "\", \"radius\": " << Num(f.radius)
         << ", \"measured_radius\": " << Num(analyses[j].radial->Radius())
         << ", \"gravity\": " << Num(analyses[j].gravity) << "}";
    }
    os << "]";
  }

 private:
  std::unique_ptr<ParGridFunction> OnSolid(const ParGridFunction& f) {
    spaces_.push_back(std::make_unique<ParFiniteElementSpace>(
        solid.get(), f.ParFESpace()->FEColl()));
    auto g = std::make_unique<ParGridFunction>(spaces_.back().get());
    ParSubMesh::Transfer(f, *g);
    return g;
  }

  CaseOptions options_;

 public:
  MeshManifest manifest;
  real_t G = 0.0, mass = 0.0, radius = 0.0, gravity = 0.0;
  int dim = 3;
  long long elements = 0;
  HYPRE_BigInt displacement_unknowns = 0, potential_unknowns = 0;
  double setup_seconds = 0.0;
  Array<int> solid_attributes, fluid_attributes;
  std::unique_ptr<ParMesh> parent;
  std::unique_ptr<ParSubMesh> solid;
  std::unique_ptr<ParGridFunction> rho, kappa, mu;
  std::unique_ptr<ParFiniteElementSpace> fes_u, fes_phi;
  std::unique_ptr<Problem> problem;
  std::vector<InterfaceAnalysis> analyses;
  int surface = -1;
  std::unique_ptr<HarmonicExpansionCoefficient> sigma, psi;

 private:
  std::vector<std::unique_ptr<ParFiniteElementSpace>> spaces_;
  std::unique_ptr<ParGridFunction> rho_solid_, kappa_solid_, mu_solid_;
  std::unique_ptr<GridFunctionCoefficient> rho_c_, kappa_c_, mu_c_,
      rho_parent_c_;
  Vector fluid_side_;
  std::vector<std::unique_ptr<ConstantCoefficient>> cmb_rho_;
  Array<int> gauge_marker_;
  std::unique_ptr<PWConstCoefficient> rho_interface_c_;
  ConstantCoefficient zero_{0.0};
  std::unique_ptr<H1_FECollection> fec_;
  std::unique_ptr<IsotropicElasticRheology> rheology_;
};

}  // namespace benchmark
