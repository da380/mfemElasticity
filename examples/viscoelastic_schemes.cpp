// ============================================================================
// viscoelastic_schemes.cpp
//
// Cost and accuracy of the time integrators of ViscoelasticOperator on one
// problem: a clamped beam (data/beam-quad.mesh: attribute 1 clamped,
// attribute 2 pulled) with a Maxwell or standard-linear-solid rheology and a
// pull varying on the relaxation-time scale, p(t) = p0 (1 + 0.5 sin(2 t /
// tau)). Every scheme is run from t = 0 to t_final for a list of step sizes
// and compared with a reference solution (RK4 at a very small step, which is
// stable there); the table reports, per run,
//
//   error   relative max-norm error of the internal variables at t_final
//   u-error relative max-norm error of the displacement at t_final
//   solves  elastic solves (the cost unit: one linear system each)
//   asm     operator assemblies (a change of effective modulus)
//   pc      preconditioner setups (the expensive part of an assembly)
//   its     CG iterations accumulated over the run
//   time    wall-clock seconds
//
// Schemes: ETD1 (exponential Euler, first order, one solve per step),
// ExpTrap (exponential trapezoid, second order, one solve per step, exact
// for a strain linear in time), BE (backward Euler through MFEM's
// ODESolver, one solve per step), SDIRK23 (MFEM's L-stable two-stage
// variant, second order, two solves per step), RK4 (explicit, four solves
// per step, only stable for dt < ~2.8 tau_min), and the adaptive exponential
// trapezoid at a few tolerances (its dt column is the number of accepted
// steps). With -gamma > 0 the Maxwell branch carries the power-law
// relaxation of Crawford et al. (2017): the times depend on the stress, the
// trapezoid and implicit schemes run their corrector, and "ExpTrap/conv"
// iterates the corrector to convergence. The explicit schemes' stability
// limit then involves the effective (shorter) times, and a run that blows
// up is reported as unstable.
//
// With -targets (a list of relative errors of the final displacement) a
// second table answers the practical question: to reach a given accuracy at
// t_final, what does each scheme cost? For every target and scheme the
// step is halved (from one step per tau) until the error is met, and the
// cost at that step is reported; for the adaptive solver the tolerance is
// tightened by factors of 2 from 0.1 until the target is met.
//
// Sample runs:
//    ./viscoelastic_schemes
//    ./viscoelastic_schemes -o 1 -r 2 -tf 8
//    ./viscoelastic_schemes -mu-inf 0.5
//    ./viscoelastic_schemes -gamma 5
//    ./viscoelastic_schemes -targets 1e-2,1e-3,1e-4
//    ./viscoelastic_schemes -gamma 5 -targets 1e-2,1e-3
//    ./viscoelastic_schemes -tau-ratio 100 -targets 1e-2,1e-3,1e-4
//
// -tau-ratio r adds a second Maxwell branch with the relaxation time tau/r:
// the stiff case, where the explicit schemes are bound to dt < 2.8 tau/r
// while the exponential and implicit ones are not.
//
// What the tables say on the beam at order 2 (solves to reach a relative
// error of 1e-2 / 1e-3 / 1e-4 in the final displacement):
//   one relaxation time, smooth forcing: RK4 17/17/33, ExpTrap 5/17/65,
//     SDIRK23 9/17/65, BE 16/256/2048, ETD1 257/2049/16385 — nothing is
//     stiff, so the explicit scheme is cheapest and the first-order schemes
//     are hopeless;
//   two relaxation times, ratio 100: RK4 1025 for any target (stability),
//     SDIRK23 17/65/129, ExpTrap 17/129/513, adaptive ExpTrap 47/66/123,
//     BE 16/256/2048;
//   power law, gamma = 5: SDIRK23 33/65, RK4 65/65, ExpTrap 33/129,
//     BE 128/1024.
// So: for a stiff or nonlinear body the implicit and exponential schemes
// pay for themselves, MFEM's L-stable SDIRK23 (two solves per step) being
// the best fixed-step choice and the adaptive trapezoid the best at tight
// tolerances after a transient; the exponential trapezoid keeps its edge
// only where the strain is close to linear over a step.
// ============================================================================

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

namespace {

struct Cost {
  int solves = 0, assemblies = 0, setups = 0;
  long its = 0;
  double seconds = 0.0;
};

struct Counters {
  int solves, assemblies, setups;
  long its;
  explicit Counters(const LinearQuasiStaticProblemBase& p)
      : solves(p.NumSolves()),
        assemblies(p.NumAssemblies()),
        setups(p.NumPreconditionerSetups()),
        its(p.TotalIterations()) {}
};

bool IsFinite(const Vector& v) {
  for (int i = 0; i < v.Size(); i++) {
    if (!std::isfinite(v[i])) {
      return false;
    }
  }
  return true;
}

double RelMaxDiff(const Vector& a, const Vector& b) {
  Vector d(a);
  d -= b;
  return d.Normlinf() / (b.Normlinf() + 1e-300);
}

}  // namespace

int main(int argc, char* argv[]) {
  const char* mesh_file = "../data/beam-quad.mesh";
  int order = 2;
  int ref_levels = 1;
  real_t tau0 = 1.0;
  real_t mu_inf0 = 0.0;
  real_t gamma0 = 0.0;
  real_t tau_ratio = 1.0;
  real_t t_final = 4.0;
  real_t p0 = 0.05;
  int n_ref = 400;  // reference RK4 steps
  const char* steps_arg = "1,2,4,8,16";
  const char* targets_arg = "";
  int max_steps_per_tau = 4096;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref_levels, "-r", "--refinement", "Uniform refinements.");
  args.AddOption(&tau0, "-tau", "--relaxation-time", "Relaxation time.");
  args.AddOption(&mu_inf0, "-mu-inf", "--long-term-modulus",
                 "Long-term shear modulus (0: Maxwell body).");
  args.AddOption(&gamma0, "-gamma", "--power-law-gamma",
                 "Power-law nonlinearity of the relaxation (0: linear).");
  args.AddOption(&tau_ratio, "-tau-ratio", "--relaxation-time-ratio",
                 "> 1: a second branch, half the relaxable modulus, with the "
                 "relaxation time tau / ratio (a stiff problem).");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time (in units of tau).");
  args.AddOption(&p0, "-p0", "--load", "Amplitude of the pull.");
  args.AddOption(&n_ref, "-nref", "--reference-steps",
                 "RK4 steps of the reference solution.");
  args.AddOption(&steps_arg, "-steps", "--steps-per-tau",
                 "Comma-separated list of steps per relaxation time.");
  args.AddOption(&targets_arg, "-targets", "--target-errors",
                 "Comma-separated target relative errors of the final "
                 "displacement for the cost-to-tolerance table (empty: "
                 "none).");
  args.AddOption(&max_steps_per_tau, "-kmax", "--max-steps-per-tau",
                 "Give up on a target beyond this many steps per tau.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  Mesh mesh(mesh_file, 1, 1);
  const int dim = mesh.Dimension();
  for (int l = 0; l < ref_levels; l++) {
    mesh.UniformRefinement();
  }
  MFEM_VERIFY(mesh.bdr_attributes.Max() >= 2,
              "The mesh needs boundary attributes 1 (clamped) and 2 (pulled).");
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);

  // Material: kappa such that lambda = mu = 1 in the unrelaxed state.
  const bool stiff = tau_ratio > 1.0;
  ConstantCoefficient kappa(1.0 + 2.0 / dim), mu_inf(mu_inf0),
      mu1((1.0 - mu_inf0) * (stiff ? 0.5 : 1.0)), mu2(0.5 * (1.0 - mu_inf0)),
      tau(tau0), tau2(tau0 / tau_ratio), gamma_c(gamma0), n_c(3.0),
      mu0_c(1.0);
  PowerLawRelaxation law(gamma_c, n_c, mu0_c);
  std::vector<MaxwellBranch> branches{
      {&mu1, &tau, gamma0 > 0.0 ? &law : nullptr}};
  if (stiff) {
    branches.push_back({&mu2, &tau2, gamma0 > 0.0 ? &law : nullptr});
  }
  IsotropicMaxwellRheology rheology(dim, kappa, mu_inf, branches);
  const real_t tau_min = stiff ? tau0 / tau_ratio : tau0;

  // Loads: attribute 1 clamped, a pull on attribute 2 varying on the tau
  // scale.
  VectorFunctionCoefficient pull(
      dim, [p0, tau0](const Vector& /*x*/, real_t t, Vector& f) {
        f = 0.0;
        f[f.Size() - 1] = -p0 * (1.0 + 0.5 * std::sin(2.0 * t / tau0));
      });
  Array<int> ess_bdr(mesh.bdr_attributes.Max()), marker(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[0] = 1;
  marker = 0;
  marker[1] = 1;

  cout << "Mesh: " << mesh.GetNE() << " elements, " << fes.GetTrueVSize()
       << " displacement unknowns, rheology "
       << (rheology.IsLinear() ? "linear" : "power law (gamma = ")
       << (rheology.IsLinear() ? "" : std::to_string(gamma0) + ")") << "\n";

  // One problem per run, so that the counters and warm starts are clean.
  auto run = [&](const std::string& name, ODESolver& ode, int n_steps,
                 bool adaptive, real_t rtol, Vector& m_out, Vector& u_out,
                 Cost& cost, int& steps_taken, int corrector = 1) {
    LinearQuasiStaticClampedProblem problem(&fes, rheology, ess_bdr, pull,
                                            marker);
    ViscoelasticOperator visco(problem);
    visco.SetCorrectorIterations(corrector, 1e-3);
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    real_t t = 0.0;
    real_t dt = t_final * tau0 / n_steps;
    const Counters c0(problem);
    const auto w0 = chrono::steady_clock::now();
    if (adaptive) {
      auto* ad = dynamic_cast<AdaptiveExponentialTrapezoidSolver*>(&ode);
      ad->SetTolerances(rtol, 1e-12);
      steps_taken = ad->Integrate(m, t, t_final * tau0, dt);
    } else {
      for (int s = 0; s < n_steps; s++) {
        ode.Step(m, t, dt);
      }
      steps_taken = n_steps;
    }
    const bool ok = visco.SolveElastic(m, t);
    const auto w1 = chrono::steady_clock::now();
    const Counters c1(problem);
    cost.solves = c1.solves - c0.solves;
    cost.assemblies = c1.assemblies - c0.assemblies;
    cost.setups = c1.setups - c0.setups;
    cost.its = c1.its - c0.its;
    cost.seconds = chrono::duration<double>(w1 - w0).count();
    m_out = m;
    problem.Displacement().GetTrueDofs(u_out);
    if (!ok || !IsFinite(m) || !IsFinite(u_out)) {
      m_out.SetSize(1);
      m_out = infinity();
    }
    (void)name;
  };

  // Reference: RK4 at a small step (explicit, so every step costs four
  // solves; stable since dt << tau_min).
  Vector m_ref, u_ref;
  {
    RK4Solver rk4;
    Cost c;
    int n;
    if (stiff) {
      n_ref = std::max(n_ref, static_cast<int>(std::round(
                                  100.0 * t_final * tau0 / tau_min)));
    }
    run("reference", rk4, n_ref, false, 0.0, m_ref, u_ref, c, n);
    cout << "Reference: RK4, " << n_ref << " steps, " << c.solves
         << " solves, " << std::fixed << std::setprecision(2) << c.seconds
         << " s\n\n";
  }

  auto parse_list = [](const char* arg) {
    std::vector<double> out;
    std::string s(arg);
    size_t pos = 0;
    while (pos < s.size()) {
      size_t next = s.find(',', pos);
      if (next == std::string::npos) {
        next = s.size();
      }
      if (next > pos) {
        out.push_back(std::stod(s.substr(pos, next - pos)));
      }
      pos = next + 1;
    }
    return out;
  };
  std::vector<int> per_tau;
  for (double v : parse_list(steps_arg)) {
    per_tau.push_back(static_cast<int>(std::round(v)));
  }
  const std::vector<double> targets = parse_list(targets_arg);

  cout << std::left << std::setw(14) << "scheme" << std::right << std::setw(8)
       << "dt/tau" << std::setw(11) << "error" << std::setw(11) << "u-error"
       << std::setw(8) << "solves" << std::setw(6) << "asm" << std::setw(5)
       << "pc" << std::setw(8) << "its" << std::setw(9) << "time\n";
  auto report = [&](const std::string& name, const std::string& dt_label,
                    const Vector& m, const Vector& u, const Cost& c) {
    cout << std::left << std::setw(14) << name << std::right << std::setw(8)
         << dt_label;
    if (m.Size() > 1 && IsFinite(m)) {
      cout << std::setw(11) << std::scientific << std::setprecision(2)
           << RelMaxDiff(m, m_ref) << std::setw(11) << RelMaxDiff(u, u_ref);
    } else {
      cout << std::setw(11) << "unstable" << std::setw(11) << "-";
    }
    cout << std::setw(8) << c.solves << std::setw(6) << c.assemblies
         << std::setw(5) << c.setups << std::setw(8) << c.its << std::setw(9)
         << std::fixed << std::setprecision(2) << c.seconds << "\n";
  };

  struct Scheme {
    std::string name;
    std::function<std::unique_ptr<ODESolver>()> make;
    int corrector;
    bool explicit_scheme;
  };
  std::vector<Scheme> schemes;
  schemes.push_back({"ETD1", [] { return std::make_unique<ExponentialEulerSolver>(); }, 1, false});
  schemes.push_back({"ExpTrap", [] { return std::make_unique<ExponentialTrapezoidSolver>(); }, 1, false});
  if (!rheology.IsLinear()) {
    schemes.push_back({"ExpTrap/conv", [] { return std::make_unique<ExponentialTrapezoidSolver>(); }, 20, false});
  }
  schemes.push_back({"BE", [] { return std::make_unique<BackwardEulerSolver>(); }, 1, false});
  schemes.push_back({"SDIRK23", [] { return std::make_unique<SDIRK23Solver>(2); }, 1, false});
  schemes.push_back({"RK4", [] { return std::make_unique<RK4Solver>(); }, 1, true});

  for (const auto& sc : schemes) {
    for (int k : per_tau) {
      const int n_steps = static_cast<int>(std::round(t_final * k));
      const real_t dt = t_final * tau0 / n_steps;
      if (sc.explicit_scheme && dt > 2.8 * tau_min) {
        cout << std::left << std::setw(14) << sc.name << std::right
             << std::setw(8) << ("1/" + std::to_string(k))
             << "   skipped: dt > 2.8 tau_min (unstable)\n";
        continue;
      }
      auto ode = sc.make();
      Vector m, u;
      Cost c;
      int n;
      run(sc.name, *ode, n_steps, false, 0.0, m, u, c, n, sc.corrector);
      report(sc.name, "1/" + std::to_string(k), m, u, c);
    }
  }
  for (real_t rtol : {1e-2, 1e-3, 1e-4}) {
    AdaptiveExponentialTrapezoidSolver ode;
    Vector m, u;
    Cost c;
    int n;
    run("Adaptive", ode, 4, true, rtol, m, u, c, n);
    std::ostringstream label;
    label << n << " st";
    report("Adaptive " + std::to_string(rtol).substr(0, 6), label.str(), m, u,
           c);
  }
  cout << "\nThe reference is RK4 at dt = tau/" << n_ref / t_final
       << "; the adaptive rows' dt column is the number of accepted steps at "
          "the given rtol.\n";

  // --- Cost to reach a target accuracy ------------------------------------
  if (!targets.empty()) {
    cout << "\nCost to reach a target relative error of the final "
            "displacement (coarsest step, or loosest rtol, that meets it):\n"
         << std::left << std::setw(14) << "scheme" << std::right
         << std::setw(9) << "target" << std::setw(9) << "dt/tau"
         << std::setw(11) << "u-error" << std::setw(8) << "solves"
         << std::setw(6) << "asm" << std::setw(5) << "pc" << std::setw(8)
         << "its" << std::setw(9) << "time\n";
    auto row = [&](const std::string& name, double target,
                   const std::string& dt_label, double err, const Cost& c) {
      cout << std::left << std::setw(14) << name << std::right << std::setw(9)
           << std::scientific << std::setprecision(0) << target
           << std::setw(9) << dt_label;
      if (err >= 0.0) {
        cout << std::setw(11) << std::scientific << std::setprecision(2)
             << err << std::setw(8) << c.solves << std::setw(6)
             << c.assemblies << std::setw(5) << c.setups << std::setw(8)
             << c.its << std::setw(9) << std::fixed << std::setprecision(2)
             << c.seconds << "\n";
      } else {
        cout << "   not reached within " << max_steps_per_tau
             << " steps per tau\n";
      }
    };
    for (double target : targets) {
      for (const auto& sc : schemes) {
        bool found = false;
        for (int k = 1; k <= max_steps_per_tau; k *= 2) {
          const int n_steps = static_cast<int>(std::round(t_final * k));
          const real_t dt = t_final * tau0 / n_steps;
          if (sc.explicit_scheme && dt > 2.8 * tau_min) {
            continue;
          }
          auto ode = sc.make();
          Vector m, u;
          Cost c;
          int n;
          run(sc.name, *ode, n_steps, false, 0.0, m, u, c, n, sc.corrector);
          if (m.Size() > 1 && IsFinite(m)) {
            const double err = RelMaxDiff(u, u_ref);
            if (err <= target) {
              row(sc.name, target, "1/" + std::to_string(k), err, c);
              found = true;
              break;
            }
          }
        }
        if (!found) {
          row(sc.name, target, "-", -1.0, Cost());
        }
      }
      {
        bool found = false;
        for (double rtol = 0.1; rtol > 1e-8; rtol *= 0.5) {
          AdaptiveExponentialTrapezoidSolver ode;
          Vector m, u;
          Cost c;
          int n;
          run("Adaptive", ode, 4, true, rtol, m, u, c, n);
          const double err = RelMaxDiff(u, u_ref);
          if (err <= target) {
            std::ostringstream label;
            label << n << " st";
            std::ostringstream name;
            name << "Adaptive " << std::scientific << std::setprecision(0)
                 << rtol;
            row(name.str(), target, label.str(), err, c);
            found = true;
            break;
          }
        }
        if (!found) {
          row("Adaptive", target, "-", -1.0, Cost());
        }
      }
      cout << "\n";
    }
  }
  return 0;
}
