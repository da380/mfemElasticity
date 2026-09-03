#pragma once

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

class Nondimensionalisation {
 private:
  real_t L;    // Length scale [m]
  real_t T;    // Time scale [s]
  real_t RHO;  // Density scale [kg/m^3]

 public:
  Nondimensionalisation(real_t length_scale, real_t time_scale,
                        real_t density_scale)
      : L(length_scale), T(time_scale), RHO(density_scale) {}

  // Accessors
  real_t Length() const { return L; }
  real_t Time() const { return T; }
  real_t Density() const { return RHO; }

  // Derived scales
  real_t Velocity() const { return L / T; }
  real_t Acceleration() const { return L / (T * T); }
  real_t Pressure() const { return RHO * L * L / (T * T); }  // [Pa]
  real_t Gravity() const { return L / (T * T); }
  real_t Potential() const { return L * L / (T * T); }

  // Scaling functions for scalars
  real_t ScaleLength(real_t x) const { return x / L; }
  real_t UnscaleLength(real_t x_nd) const { return x_nd * L; }

  real_t ScaleDensity(real_t rho) const { return rho / RHO; }
  real_t UnscaleDensity(real_t rho_nd) const { return rho_nd * RHO; }

  real_t ScaleGravityPotential(real_t phi) const { return phi / Potential(); }
  real_t UnscaleGravityPotential(real_t phi_nd) const {
    return phi_nd * Potential();
  }

  real_t ScaleStress(real_t sigma) const { return sigma / Pressure(); }
  real_t UnscaleStress(real_t sigma_nd) const { return sigma_nd * Pressure(); }

  // Scaling for GridFunction fields
  void UnscaleGravityPotential(GridFunction &phi_gf) const {
    phi_gf *= Potential();
  }
  void UnscaleDisplacement(GridFunction &u_gf) const { u_gf *= L; }
  void UnscaleStress(GridFunction &sigma_gf) const { sigma_gf *= Pressure(); }

  // Create a scaled density coefficient from a dimensional one
  Coefficient *MakeScaledDensityCoefficient(Coefficient &rho_coeff) const {
    return new ProductCoefficient(1.0 / RHO, rho_coeff);
  }

  void Print() const {
    cout << "Scaling parameters:\n";
    cout << "  Length scale: " << L << " m\n";
    cout << "  Time scale: " << T << " s\n";
    cout << "  Density scale: " << RHO << " kg/m^3\n";
    cout << "  Gravity potential scale: " << Potential() << " m^2/s^2\n";
  }
};

struct Constants {
 public:
  static constexpr real_t G = 6.6743e-11;
  static constexpr real_t c = 2.99792458e8;
  static constexpr real_t h = 6.62607015e-34;
  static constexpr real_t _h = 1.054571817e-34;
  static constexpr real_t kB = 1.380649e-23;
  static constexpr real_t NA = 6.02214076e23;
  static constexpr real_t e = 1.602176634e-19;
  static constexpr real_t epi0 = 8.854187817e-12;
  static constexpr real_t mu0 = 1.25663706212e-6;

  static constexpr real_t R = 6371e3;
};
