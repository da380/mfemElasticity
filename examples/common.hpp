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

class TwoBlockRigidBodySolver : public Solver {
 private:
  FiniteElementSpace *_fes_u;
  FiniteElementSpace *_fes_phi;
  Array<int> *_block_offsets;
  VectorCoefficient *_dphi0_coeff;

  std::vector<BlockVector *> _ns;

  Solver *_solver = nullptr;

  mutable BlockVector _b;
  mutable BlockVector _x;

  real_t Dot(const Vector &x, const Vector &y) const {
    return InnerProduct(x, y);
  }

  real_t Norm(const Vector &x) const { return std::sqrt(Dot(x, x)); }

  real_t BlockDot(const BlockVector &x, const BlockVector &y) const {
    return InnerProduct(x.GetBlock(0), y.GetBlock(0)) +
           InnerProduct(x.GetBlock(1), y.GetBlock(1));
  }

  real_t BlockNorm(const BlockVector &x) const {
    return std::sqrt(std::max(BlockDot(x, x), real_t(0.0)));
  }

  int ElasticRigidBodyDim() const {
    int dim = _fes_u->GetVDim();
    return dim * (dim + 1) / 2;
  }

  void AddCoupledRigidMode(VectorCoefficient &u_coeff) {
    GridFunction u_gf(_fes_u);
    GridFunction phi_gf(_fes_phi);

    u_gf = 0.0;
    phi_gf = 0.0;

    u_gf.ProjectCoefficient(u_coeff);

    InnerProductCoefficient phi_coeff(u_coeff, *_dphi0_coeff);

    phi_gf.ProjectCoefficient(phi_coeff);
    phi_gf.Neg();

    BlockVector *nv = new BlockVector(*_block_offsets);
    *nv = 0.0;

    u_gf.GetTrueDofs(nv->GetBlock(0));
    phi_gf.GetTrueDofs(nv->GetBlock(1));

    _ns.push_back(nv);
  }

  void AddPurePhiConstantMode() {
    GridFunction phi_gf(_fes_phi);
    phi_gf = 1.0;

    BlockVector *nv = new BlockVector(*_block_offsets);
    *nv = 0.0;

    phi_gf.GetTrueDofs(nv->GetBlock(1));

    _ns.push_back(nv);
  }

  void GramSchmidt() {
    for (int i = 0; i < (int)_ns.size(); i++) {
      BlockVector &nv1 = *_ns[i];

      for (int j = 0; j < i; j++) {
        BlockVector &nv2 = *_ns[j];

        real_t product = BlockDot(nv1, nv2);
        nv1.Add(-product, nv2);
      }

      real_t norm = BlockNorm(nv1);

      MFEM_VERIFY(norm > 0.0,
                  "zero nullspace vector in TwoBlockRigidBodySolver");

      nv1 /= norm;
    }
  }

 public:
  void ProjectOrthogonalToNullspace(const Vector &x, Vector &y) const {
    y = x;

    Vector y_u(y.GetData() + (*_block_offsets)[0],
               (*_block_offsets)[1] - (*_block_offsets)[0]);

    Vector y_phi(y.GetData() + (*_block_offsets)[1],
                 (*_block_offsets)[2] - (*_block_offsets)[1]);

    for (int i = 0; i < (int)_ns.size(); i++) {
      const BlockVector &nv = *_ns[i];

      real_t product = InnerProduct(y_u, nv.GetBlock(0)) +
                       InnerProduct(y_phi, nv.GetBlock(1));

      y.Add(-product, nv);
    }
  }

 public:
  TwoBlockRigidBodySolver(FiniteElementSpace *fes_u,
                          FiniteElementSpace *fes_phi,
                          Array<int> *block_offsets,
                          VectorCoefficient *dphi0_coeff)
      : Solver((*block_offsets)[2], false),
        _fes_u(fes_u),
        _fes_phi(fes_phi),
        _block_offsets(block_offsets),
        _dphi0_coeff(dphi0_coeff),
        _b(*block_offsets),
        _x(*block_offsets) {
    int dim = _fes_u->GetVDim();

    MFEM_ASSERT(dim == 2 || dim == 3, "dimensions must be two or three");

    height = (*_block_offsets)[2];
    width = height;

    for (int component = 0; component < dim; component++) {
      RigidTranslation u_coeff(dim, component);
      AddCoupledRigidMode(u_coeff);
    }

    if (dim == 2) {
      RigidRotation u_coeff(dim, 2);
      AddCoupledRigidMode(u_coeff);

      AddPurePhiConstantMode();
    } else {
      for (int component = 0; component < dim; component++) {
        RigidRotation u_coeff(dim, component);
        AddCoupledRigidMode(u_coeff);
      }
    }

    GramSchmidt();

    cout << "Block nullspace dimension = " << _ns.size() << endl;
  }

  ~TwoBlockRigidBodySolver() {
    for (int i = 0; i < (int)_ns.size(); i++) {
      delete _ns[i];
    }
  }

  void SetSolver(Solver &solver) {
    _solver = &solver;

    height = _solver->Height();
    width = _solver->Width();

    MFEM_VERIFY(height == width, "solver must be square");
  }

  void SetOperator(const Operator &op) override {
    MFEM_VERIFY(_solver != nullptr, "SetSolver must be called first");

    _solver->SetOperator(op);

    height = _solver->Height();
    width = _solver->Width();

    MFEM_VERIFY(height == width, "solver must be square");
  }

  void Mult(const Vector &b, Vector &x) const override {
    MFEM_VERIFY(_solver != nullptr, "SetSolver must be called first");

    ProjectOrthogonalToNullspace(b, _b);

    _x = 0.0;
    _solver->Mult(_b, _x);

    ProjectOrthogonalToNullspace(_x, x);
  }
};

#ifdef MFEM_USE_MPI
class TwoBlockRigidBodySolverParallel : public Solver {
 private:
  MPI_Comm _comm;

  ParFiniteElementSpace *_fes_u;
  ParFiniteElementSpace *_fes_phi;

  Array<int> *_block_true_offsets;

  VectorCoefficient *_dphi0_coeff;

  std::vector<std::unique_ptr<BlockVector>> _ns;

  Solver *_solver = nullptr;

  mutable BlockVector _b;
  mutable BlockVector _x;

  bool _add_constant_phi_mode;

  real_t Dot(const Vector &x, const Vector &y) const {
    return InnerProduct(_comm, x, y);
  }

  real_t Norm(const Vector &x) const { return std::sqrt(Dot(x, x)); }

  real_t BlockDot(const BlockVector &x, const BlockVector &y) const {
    return InnerProduct(_comm, x.GetBlock(0), y.GetBlock(0)) +
           InnerProduct(_comm, x.GetBlock(1), y.GetBlock(1));
  }

  real_t BlockNorm(const BlockVector &x) const {
    return std::sqrt(std::max(BlockDot(x, x), real_t(0.0)));
  }

  void AddCoupledRigidMode(VectorCoefficient &u_coeff) {
    ParGridFunction u_gf(_fes_u);
    ParGridFunction phi_gf(_fes_phi);

    u_gf = 0.0;
    phi_gf = 0.0;

    u_gf.ProjectCoefficient(u_coeff);

    InnerProductCoefficient phi_coeff(u_coeff, *_dphi0_coeff);

    phi_gf.ProjectCoefficient(phi_coeff);
    phi_gf.Neg();

    auto nv = std::make_unique<BlockVector>(*_block_true_offsets);
    *nv = 0.0;

    u_gf.GetTrueDofs(nv->GetBlock(0));
    phi_gf.GetTrueDofs(nv->GetBlock(1));

    _ns.push_back(std::move(nv));
  }

  void AddPurePhiConstantMode() {
    ParGridFunction phi_gf(_fes_phi);
    phi_gf = 1.0;

    auto nv = std::make_unique<BlockVector>(*_block_true_offsets);
    *nv = 0.0;

    phi_gf.GetTrueDofs(nv->GetBlock(1));

    _ns.push_back(std::move(nv));
  }

  void GramSchmidt() {
    for (int i = 0; i < (int)_ns.size(); i++) {
      BlockVector &ni = *_ns[i];

      for (int j = 0; j < i; j++) {
        BlockVector &nj = *_ns[j];

        real_t product = BlockDot(ni, nj);
        ni.Add(-product, nj);
      }

      real_t norm = BlockNorm(ni);

      MFEM_VERIFY(norm > 0.0,
                  "zero nullspace vector in TwoBlockRigidBodySolverParallel");

      ni /= norm;
    }
  }

  void ProjectOrthogonalToNullspace(const Vector &x, Vector &y) const {
    y = x;

    Vector y_u(y.GetData() + (*_block_true_offsets)[0],
               (*_block_true_offsets)[1] - (*_block_true_offsets)[0]);

    Vector y_phi(y.GetData() + (*_block_true_offsets)[1],
                 (*_block_true_offsets)[2] - (*_block_true_offsets)[1]);

    for (int i = 0; i < (int)_ns.size(); i++) {
      const BlockVector &ni = *_ns[i];

      real_t product = InnerProduct(_comm, y_u, ni.GetBlock(0)) +
                       InnerProduct(_comm, y_phi, ni.GetBlock(1));

      y.Add(-product, ni);
    }
  }

 public:
  TwoBlockRigidBodySolverParallel(MPI_Comm comm, ParFiniteElementSpace *fes_u,
                                  ParFiniteElementSpace *fes_phi,
                                  Array<int> *block_true_offsets,
                                  VectorCoefficient *dphi0_coeff,
                                  bool add_constant_phi_mode)
      : Solver(0, false),
        _comm(comm),
        _fes_u(fes_u),
        _fes_phi(fes_phi),
        _block_true_offsets(block_true_offsets),
        _dphi0_coeff(dphi0_coeff),
        _b(*block_true_offsets),
        _x(*block_true_offsets),
        _add_constant_phi_mode(add_constant_phi_mode) {
    int dim = _fes_u->GetVDim();

    MFEM_ASSERT(dim == 2 || dim == 3, "dimension must be 2 or 3");

    height = (*_block_true_offsets)[2];
    width = height;

    for (int c = 0; c < dim; c++) {
      RigidTranslation u_coeff(dim, c);
      AddCoupledRigidMode(u_coeff);
    }

    if (dim == 2) {
      RigidRotation u_coeff(dim, 2);
      AddCoupledRigidMode(u_coeff);
    } else {
      for (int c = 0; c < dim; c++) {
        RigidRotation u_coeff(dim, c);
        AddCoupledRigidMode(u_coeff);
      }
    }

    if (_add_constant_phi_mode) {
      AddPurePhiConstantMode();
    }

    GramSchmidt();

    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank == 0) {
      cout << "Parallel block nullspace dimension = " << _ns.size() << endl;
    }
  }

  void SetSolver(Solver &solver) {
    _solver = &solver;

    height = _solver->Height();
    width = _solver->Width();

    MFEM_VERIFY(height == width, "solver must be square");
  }

  void SetOperator(const Operator &op) override {
    MFEM_VERIFY(_solver != nullptr, "SetSolver must be called first");

    _solver->SetOperator(op);

    height = _solver->Height();
    width = _solver->Width();

    MFEM_VERIFY(height == width, "solver must be square");
  }

  void Mult(const Vector &b, Vector &x) const override {
    MFEM_VERIFY(_solver != nullptr, "SetSolver must be called first");

    ProjectOrthogonalToNullspace(b, _b);

    _x = 0.0;

    _solver->iterative_mode = iterative_mode;
    _solver->Mult(_b, _x);

    ProjectOrthogonalToNullspace(_x, x);
  }
};
#endif  // MFEM_USE_MPI

#ifdef MFEM_USE_MPI
#endif  // MFEM_USE_MPI

#ifdef MFEM_USE_MPI
void SendParSolutionToGLVis(MPI_Comm world, int myid, int dim,
                            mfem::ParMesh &pm, mfem::ParGridFunction &gf,
                            const char *title) {
  const int active = (pm.GetNE() > 0) ? 1 : 0;

  int active_count = 0;
  MPI_Allreduce(&active, &active_count, 1, MPI_INT, MPI_SUM, world);

  if (active_count == 0) {
    if (myid == 0) {
      std::cout << "Skipping GLVis output for empty mesh: " << title
                << std::endl;
    }
    return;
  }

  MPI_Comm vis_comm;
  MPI_Comm_split(world, active ? 0 : MPI_UNDEFINED, myid, &vis_comm);

  if (!active) {
    MPI_Barrier(world);
    return;
  }

  int vis_rank = 0;
  int vis_size = 1;

  MPI_Comm_rank(vis_comm, &vis_rank);
  MPI_Comm_size(vis_comm, &vis_size);

  char vishost[] = "localhost";
  int visport = 19916;

  mfem::socketstream sock(vishost, visport);

  sock << "parallel " << vis_size << " " << vis_rank << "\n";

  sock.precision(8);

  sock << "solution\n"
       << pm << gf << "window_title '" << title << "'" << std::endl;

  if (dim == 2) {
    sock << "keys Rjlbc\n" << std::flush;
  } else {
    sock << "keys RRRilc\n" << std::flush;
  }

  MPI_Comm_free(&vis_comm);

  MPI_Barrier(world);
}
#endif  // MFEM_USE_MPI
