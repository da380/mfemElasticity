/**
 * @file referential_problem.cpp
 * @brief Implementation of ReferentialElasticRheology and
 * LinearQuasiStaticReferentialProblem.
 */

#include "mfemElasticity/referential_problem.hpp"

#include <cmath>
#include <numbers>

#include "mfemElasticity/detail/fem_factory.hpp"
#include "mfemElasticity/mesh.hpp"

namespace mfemElasticity {

using namespace mfem;

namespace {
constexpr real_t kPi = std::numbers::pi_v<real_t>;

/// Stiffness of the referential rheology: the total-Lagrangian split,
/// fixed integrators, weights no-ops.
class ReferentialStiffness : public ElasticStiffness {
 public:
  ReferentialStiffness(MatrixCoefficient& C, MatrixCoefficient& S,
                       Diffeomorphism& map)
      : C_(&C), S_(&S), map_(&map) {}

  void AddIntegrators(BilinearForm& form, Array<int>* marker) override {
    auto add = [&](BilinearFormIntegrator* i) {
      if (marker) {
        form.AddDomainIntegrator(i, *marker);
      } else {
        form.AddDomainIntegrator(i);
      }
    };
    add(new MaterialStiffnessIntegrator(*C_, *map_));
    add(new GeometricStiffnessIntegrator(*S_));
  }

  void SetRelaxationWeights(const std::vector<Coefficient*>&) override {}
  void ClearRelaxationWeights() override {}
  bool IsRelaxed() const override { return false; }

 private:
  MatrixCoefficient* C_;
  MatrixCoefficient* S_;
  Diffeomorphism* map_;
};

// (MappedRotation now lives in null_space.hpp.)

}  // namespace

SlipInterfaceBlocks NewSlipInterfaceMatrix(FiniteElementSpace& fes_s,
                                           const SparseMatrix& J,
                                           const Array<int>& interface_marker,
                                           Coefficient& pi,
                                           Diffeomorphism& map) {
  // The one-sided kernel G on the solid side.
  Array<int> marker(interface_marker);
  BilinearForm g(&fes_s);
  g.AddBoundaryIntegrator(new SlipInterfacePressureIntegrator(pi, map),
                          marker);
  g.Assemble();
  g.Finalize();
  const SparseMatrix& G = g.SpMat();
  std::unique_ptr<SparseMatrix> Gt(Transpose(G));

  // B = (S^T G D + D^T G^T S)/2, S = [I, J], D = [I, -J]:
  //   ss =  (G + G^T)/2,          sf = (G^T - G)/2 J,
  //   fs = sf^T,                  ff = -J^T (G + G^T)/2 J.
  std::unique_ptr<SparseMatrix> Gsym(Add(0.5, G, 0.5, *Gt));
  std::unique_ptr<SparseMatrix> Gskew(Add(0.5, *Gt, -0.5, G));
  std::unique_ptr<SparseMatrix> Jt(Transpose(J));

  // Normal convention: the derivation's N points OUT OF THE FLUID, but
  // assembly on the solid submesh supplies the solid's outward normal,
  // which is -N -- hence the overall minus on every block (pinned by
  // the discrete-vs-quadrature cross-check).
  SlipInterfaceBlocks B;
  B.ss = std::make_unique<SparseMatrix>(*Gsym);
  *B.ss *= -1.0;
  B.sf.reset(mfem::Mult(*Gskew, J));
  *B.sf *= -1.0;
  B.fs.reset(Transpose(*B.sf));
  std::unique_ptr<SparseMatrix> tmp(mfem::Mult(*Jt, *Gsym));
  B.ff.reset(mfem::Mult(*tmp, J));
  return B;
}


std::unique_ptr<mfem::SparseMatrix> NewRadialVacuumExtension(
    FiniteElementSpace& body_fes, FiniteElementSpace& buffer_fes,
    real_t r_body, real_t r_outer, real_t taper_power, real_t pullback) {
  MFEM_VERIFY(body_fes.FEColl() == buffer_fes.FEColl() &&
                  body_fes.GetVDim() == buffer_fes.GetVDim(),
              "NewRadialVacuumExtension: the spaces must share a "
              "collection and vdim.");
  Mesh* body = body_fes.GetMesh();
  Mesh* buffer = buffer_fes.GetMesh();
  const int dim = body->Dimension();

  // The shared parent and the trace pairing.
  auto* body_sub = dynamic_cast<SubMesh*>(body);
  auto* buffer_sub = dynamic_cast<SubMesh*>(buffer);
  MFEM_VERIFY(body_sub && buffer_sub &&
                  body_sub->GetParent() == buffer_sub->GetParent(),
              "NewRadialVacuumExtension: both spaces must live on SubMeshes "
              "of one parent.");
  FiniteElementSpace parent_fes(
      const_cast<Mesh*>(static_cast<const Mesh*>(body_sub->GetParent())),
      const_cast<FiniteElementCollection*>(body_fes.FEColl()),
      body_fes.GetVDim(), body_fes.GetOrdering());
  SubMeshDofInjection inj_body(body_fes, parent_fes);
  SubMeshDofInjection inj_buffer(buffer_fes, parent_fes);
  auto J = NewSubMeshPairingMatrix(inj_buffer, inj_body);  // buffer x body

  // Scalar nodal coordinates of the buffer space.
  const int ns_buf = buffer_fes.GetNDofs();
  DenseMatrix coords(dim, ns_buf);
  {
    Array<int> dofs;
    Vector x(dim);
    for (int e = 0; e < buffer->GetNE(); e++) {
      const auto* fe = buffer_fes.GetFE(e);
      auto* T = buffer->GetElementTransformation(e);
      buffer_fes.GetElementDofs(e, dofs);
      const auto& nodes = fe->GetNodes();
      for (int i = 0; i < dofs.Size(); i++) {
        T->Transform(nodes.IntPoint(i), x);
        for (int d = 0; d < dim; d++) {
          coords(d, dofs[i]) = x(d);
        }
      }
    }
  }

  // Which buffer scalar dofs are paired (the trace): from the vdof
  // pairing's rows (component 0 suffices for byNODES ordering).
  std::vector<char> paired(ns_buf, 0);
  for (int sb = 0; sb < ns_buf; sb++) {
    if (J->RowSize(buffer_fes.DofToVDof(sb, 0)) > 0) {
      paired[sb] = 1;
    }
  }

  // Locate the interior nodes' surface projections in the body mesh.
  std::vector<int> interior;
  for (int sb = 0; sb < ns_buf; sb++) {
    if (!paired[sb]) {
      interior.push_back(sb);
    }
  }
  DenseMatrix pts(dim, static_cast<int>(interior.size()));
  for (std::size_t i = 0; i < interior.size(); i++) {
    real_t r = 0.0;
    for (int d = 0; d < dim; d++) {
      r += coords(d, interior[i]) * coords(d, interior[i]);
    }
    r = std::sqrt(r);
    const real_t scale = pullback * r_body / std::max(r, real_t(1e-30));
    for (int d = 0; d < dim; d++) {
      pts(d, i) = scale * coords(d, interior[i]);
    }
  }
  Array<int> elem;
  Array<IntegrationPoint> ips;
  if (pts.Width() > 0) {
    body->FindPoints(pts, elem, ips);
    // Retry unfound projections deeper inside the body (the curved
    // discrete boundary can dip below the nominal radius); the interior
    // rule is a gauge choice, so the retreat costs nothing.
    for (real_t factor : {0.99, 0.97, 0.9}) {
      int missing = 0;
      for (int i = 0; i < elem.Size(); i++) {
        if (elem[i] < 0) {
          missing++;
        }
      }
      if (missing == 0) {
        break;
      }
      DenseMatrix retry(dim, missing);
      std::vector<int> which;
      for (int i = 0; i < elem.Size(); i++) {
        if (elem[i] < 0) {
          for (int d = 0; d < dim; d++) {
            retry(d, static_cast<int>(which.size())) =
                pts(d, i) * factor / pullback;
          }
          which.push_back(i);
        }
      }
      Array<int> elem2;
      Array<IntegrationPoint> ips2;
      body->FindPoints(retry, elem2, ips2);
      for (std::size_t j = 0; j < which.size(); j++) {
        if (elem2[j] >= 0) {
          elem[which[j]] = elem2[j];
          ips[which[j]] = ips2[j];
        }
      }
    }
  }

  auto E = std::make_unique<SparseMatrix>(buffer_fes.GetVSize(),
                                          body_fes.GetVSize());
  const int vdim = body_fes.GetVDim();
  // Trace rows: copy the body values exactly.
  {
    Array<int> cols;
    Vector vals;
    for (int sb = 0; sb < ns_buf; sb++) {
      if (!paired[sb]) {
        continue;
      }
      for (int k = 0; k < vdim; k++) {
        const int row = buffer_fes.DofToVDof(sb, k);
        J->GetRow(row, cols, vals);
        for (int j = 0; j < cols.Size(); j++) {
          E->Set(row, cols[j], vals[j]);
        }
      }
    }
  }
  // Interior rows: tapered radial interpolation.
  {
    Array<int> dofs;
    Vector shape;
    for (std::size_t i = 0; i < interior.size(); i++) {
      const int sb = interior[i];
      MFEM_VERIFY(elem[i] >= 0,
                  "NewRadialVacuumExtension: surface projection not found "
                  "in the body mesh; reduce `pullback`.");
      real_t r = 0.0;
      for (int d = 0; d < dim; d++) {
        r += coords(d, sb) * coords(d, sb);
      }
      r = std::sqrt(r);
      real_t t = (r_outer - r) / (r_outer - r_body);
      t = std::min(real_t(1), std::max(real_t(0), t));
      t = std::pow(t, taper_power);
      if (t == 0.0) {
        continue;
      }
      const auto* fe = body_fes.GetFE(elem[i]);
      shape.SetSize(fe->GetDof());
      fe->CalcShape(ips[i], shape);
      body_fes.GetElementDofs(elem[i], dofs);
      for (int k = 0; k < vdim; k++) {
        const int row = buffer_fes.DofToVDof(sb, k);
        for (int a = 0; a < dofs.Size(); a++) {
          E->Set(row, body_fes.DofToVDof(dofs[a], k), t * shape(a));
        }
      }
    }
  }
  E->Finalize();
  return E;
}


std::unique_ptr<mfem::SparseMatrix> NewRadialFluidExtension(
    FiniteElementSpace& solid_fes, FiniteElementSpace& fluid_fes,
    real_t r_interface, real_t taper_power, real_t pushout) {
  MFEM_VERIFY(solid_fes.FEColl() == fluid_fes.FEColl() &&
                  solid_fes.GetVDim() == fluid_fes.GetVDim(),
              "NewRadialFluidExtension: the spaces must share a "
              "collection and vdim.");
  Mesh* solid = solid_fes.GetMesh();
  Mesh* fluid = fluid_fes.GetMesh();
  const int dim = solid->Dimension();

  // The shared parent and the trace pairing.
  auto* solid_sub = dynamic_cast<SubMesh*>(solid);
  auto* fluid_sub = dynamic_cast<SubMesh*>(fluid);
  MFEM_VERIFY(solid_sub && fluid_sub &&
                  solid_sub->GetParent() == fluid_sub->GetParent(),
              "NewRadialFluidExtension: both spaces must live on SubMeshes "
              "of one parent.");
  FiniteElementSpace parent_fes(
      const_cast<Mesh*>(static_cast<const Mesh*>(solid_sub->GetParent())),
      const_cast<FiniteElementCollection*>(solid_fes.FEColl()),
      solid_fes.GetVDim(), solid_fes.GetOrdering());
  SubMeshDofInjection inj_solid(solid_fes, parent_fes);
  SubMeshDofInjection inj_fluid(fluid_fes, parent_fes);
  auto J = NewSubMeshPairingMatrix(inj_fluid, inj_solid);  // fluid x solid

  // Scalar nodal coordinates of the fluid space.
  const int ns_flu = fluid_fes.GetNDofs();
  DenseMatrix coords(dim, ns_flu);
  {
    Array<int> dofs;
    Vector x(dim);
    for (int e = 0; e < fluid->GetNE(); e++) {
      const auto* fe = fluid_fes.GetFE(e);
      auto* T = fluid->GetElementTransformation(e);
      fluid_fes.GetElementDofs(e, dofs);
      const auto& nodes = fe->GetNodes();
      for (int i = 0; i < dofs.Size(); i++) {
        T->Transform(nodes.IntPoint(i), x);
        for (int d = 0; d < dim; d++) {
          coords(d, dofs[i]) = x(d);
        }
      }
    }
  }

  // Which fluid scalar dofs are paired (the interface trace).
  std::vector<char> paired(ns_flu, 0);
  for (int sb = 0; sb < ns_flu; sb++) {
    if (J->RowSize(fluid_fes.DofToVDof(sb, 0)) > 0) {
      paired[sb] = 1;
    }
  }

  // Interior nodes carrying a nonzero taper. A node at (or numerically
  // at) the centre has t = 0 and an undefined radial direction, so its
  // row is simply left zero -- a valid piece of the gauge.
  std::vector<int> interior;
  std::vector<real_t> taper;
  for (int sb = 0; sb < ns_flu; sb++) {
    if (paired[sb]) {
      continue;
    }
    real_t r = 0.0;
    for (int d = 0; d < dim; d++) {
      r += coords(d, sb) * coords(d, sb);
    }
    r = std::sqrt(r);
    real_t t = r / r_interface;
    t = std::min(real_t(1), std::max(real_t(0), t));
    t = std::pow(t, taper_power);
    if (t > 0.0 && r > 1e-12 * r_interface) {
      interior.push_back(sb);
      taper.push_back(t);
    }
  }

  // Locate the interior nodes' interface projections in the solid mesh,
  // pushed slightly OUTWARD (pushout > 1): the solid lies outside the
  // interface, and its discrete inner boundary can bulge above the
  // nominal radius. The interior rule is a gauge choice, so the push
  // costs nothing.
  DenseMatrix pts(dim, static_cast<int>(interior.size()));
  for (std::size_t i = 0; i < interior.size(); i++) {
    real_t r = 0.0;
    for (int d = 0; d < dim; d++) {
      r += coords(d, interior[i]) * coords(d, interior[i]);
    }
    r = std::sqrt(r);
    const real_t scale = pushout * r_interface / r;
    for (int d = 0; d < dim; d++) {
      pts(d, i) = scale * coords(d, interior[i]);
    }
  }
  Array<int> elem;
  Array<IntegrationPoint> ips;
  if (pts.Width() > 0) {
    solid->FindPoints(pts, elem, ips);
    // Retry unfound projections deeper inside the solid (further out).
    for (real_t factor : {1.01, 1.03, 1.1}) {
      int missing = 0;
      for (int i = 0; i < elem.Size(); i++) {
        if (elem[i] < 0) {
          missing++;
        }
      }
      if (missing == 0) {
        break;
      }
      DenseMatrix retry(dim, missing);
      std::vector<int> which;
      for (int i = 0; i < elem.Size(); i++) {
        if (elem[i] < 0) {
          for (int d = 0; d < dim; d++) {
            retry(d, static_cast<int>(which.size())) =
                pts(d, i) * factor / pushout;
          }
          which.push_back(i);
        }
      }
      Array<int> elem2;
      Array<IntegrationPoint> ips2;
      solid->FindPoints(retry, elem2, ips2);
      for (std::size_t j = 0; j < which.size(); j++) {
        if (elem2[j] >= 0) {
          elem[which[j]] = elem2[j];
          ips[which[j]] = ips2[j];
        }
      }
    }
  }

  auto E = std::make_unique<SparseMatrix>(fluid_fes.GetVSize(),
                                          solid_fes.GetVSize());
  const int vdim = solid_fes.GetVDim();
  // Trace rows: copy the solid values exactly.
  {
    Array<int> cols;
    Vector vals;
    for (int sb = 0; sb < ns_flu; sb++) {
      if (!paired[sb]) {
        continue;
      }
      for (int k = 0; k < vdim; k++) {
        const int row = fluid_fes.DofToVDof(sb, k);
        J->GetRow(row, cols, vals);
        for (int j = 0; j < cols.Size(); j++) {
          E->Set(row, cols[j], vals[j]);
        }
      }
    }
  }
  // Interior rows: tapered inward radial interpolation of the solid trace.
  {
    Array<int> dofs;
    Vector shape;
    for (std::size_t i = 0; i < interior.size(); i++) {
      const int sb = interior[i];
      MFEM_VERIFY(elem[i] >= 0,
                  "NewRadialFluidExtension: interface projection not found "
                  "in the solid mesh; increase `pushout`.");
      const real_t t = taper[i];
      const auto* fe = solid_fes.GetFE(elem[i]);
      shape.SetSize(fe->GetDof());
      fe->CalcShape(ips[i], shape);
      solid_fes.GetElementDofs(elem[i], dofs);
      for (int k = 0; k < vdim; k++) {
        const int row = fluid_fes.DofToVDof(sb, k);
        for (int a = 0; a < dofs.Size(); a++) {
          E->Set(row, solid_fes.DofToVDof(dofs[a], k), t * shape(a));
        }
      }
    }
  }
  E->Finalize();
  return E;
}


#ifdef MFEM_USE_MPI
std::unique_ptr<mfem::HypreParMatrix> NewRadialFluidExtension(
    ParFiniteElementSpace& solid_fes, ParFiniteElementSpace& fluid_fes,
    real_t r_interface, real_t taper_power, real_t pushout) {
  MFEM_VERIFY(solid_fes.FEColl() == fluid_fes.FEColl() &&
                  solid_fes.GetVDim() == fluid_fes.GetVDim(),
              "NewRadialFluidExtension: the spaces must share a "
              "collection and vdim.");
  auto* solid_sub = dynamic_cast<ParSubMesh*>(solid_fes.GetParMesh());
  auto* fluid_sub = dynamic_cast<ParSubMesh*>(fluid_fes.GetParMesh());
  MFEM_VERIFY(solid_sub && fluid_sub &&
                  solid_sub->GetParent() == fluid_sub->GetParent(),
              "NewRadialFluidExtension: both spaces must live on "
              "ParSubMeshes of one parent.");
  MPI_Comm comm = solid_fes.GetComm();
  const int dim = solid_sub->Dimension();
  const int vdim = solid_fes.GetVDim();

  ParFiniteElementSpace parent_fes(
      const_cast<ParMesh*>(static_cast<const ParMesh*>(solid_sub->GetParent())),
      const_cast<FiniteElementCollection*>(solid_fes.FEColl()), vdim,
      solid_fes.GetOrdering());
  SubMeshDofInjection inj_solid(solid_fes, parent_fes);
  SubMeshDofInjection inj_fluid(fluid_fes, parent_fes);
  auto J = NewSubMeshPairingTrueDofMatrix(inj_fluid, inj_solid);

  // Paired (interface trace) rows of the owned fluid true dofs.
  const int nrows = fluid_fes.GetTrueVSize();
  std::vector<char> paired(nrows, 0);
  {
    SparseMatrix diag, offd;
    HYPRE_BigInt* cmap = nullptr;
    J->GetDiag(diag);
    J->GetOffd(offd, cmap);
    for (int t = 0; t < nrows; t++) {
      if (diag.RowSize(t) + offd.RowSize(t) > 0) {
        paired[t] = 1;
      }
    }
  }

  // Scalar nodal coordinates of the fluid space (local).
  const int ns_flu = fluid_fes.GetNDofs();
  DenseMatrix coords(dim, ns_flu);
  {
    Array<int> dofs;
    Vector x(dim);
    for (int e = 0; e < fluid_sub->GetNE(); e++) {
      const auto* fe = fluid_fes.GetFE(e);
      auto* T = fluid_sub->GetElementTransformation(e);
      fluid_fes.GetElementDofs(e, dofs);
      const auto& nodes = fe->GetNodes();
      for (int i = 0; i < dofs.Size(); i++) {
        T->Transform(nodes.IntPoint(i), x);
        for (int d = 0; d < dim; d++) {
          coords(d, dofs[i]) = x(d);
        }
      }
    }
  }

  // The owned, unpaired scalar nodes with a live taper: the queries.
  // Centre nodes (undefined radial direction) keep zero rows -- gauge.
  struct Query {
    int sdof;
    real_t t, r;
  };
  std::vector<Query> queries;
  for (int sb = 0; sb < ns_flu; sb++) {
    const int lt0 = fluid_fes.GetLocalTDofNumber(fluid_fes.DofToVDof(sb, 0));
    if (lt0 < 0 || paired[lt0]) {
      continue;
    }
    real_t r = 0.0;
    for (int d = 0; d < dim; d++) {
      r += coords(d, sb) * coords(d, sb);
    }
    r = std::sqrt(r);
    real_t t = r / r_interface;
    t = std::min(real_t(1), std::max(real_t(0), t));
    t = std::pow(t, taper_power);
    if (t == 0.0 || r <= 1e-12 * r_interface) {
      continue;
    }
    queries.push_back({sb, t, r});
  }

  // Interpolation rows, resolved over retry rounds pushing OUTWARD into
  // the solid (its discrete inner boundary can bulge above the nominal
  // radius); the query-reply exchange as in NewRadialVacuumExtension.
  std::vector<std::vector<HYPRE_BigInt>> row_cols(queries.size());
  std::vector<std::vector<real_t>> row_vals(queries.size());
  std::vector<char> resolved(queries.size(), 0);
  int ranks = 0, rank = 0;
  MPI_Comm_size(comm, &ranks);
  MPI_Comm_rank(comm, &rank);

  for (real_t factor : {1.0, 1.01, 1.03, 1.1}) {
    std::vector<real_t> my_pts;
    std::vector<int> my_qid;
    for (std::size_t q = 0; q < queries.size(); q++) {
      if (resolved[q]) {
        continue;
      }
      const int sb = queries[q].sdof;
      const real_t scale = factor * pushout * r_interface / queries[q].r;
      for (int d = 0; d < dim; d++) {
        my_pts.push_back(scale * coords(d, sb));
      }
      my_qid.push_back(static_cast<int>(q));
    }
    int my_n = static_cast<int>(my_qid.size());
    std::vector<int> counts(ranks), displs(ranks + 1, 0);
    MPI_Allgather(&my_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    long long total = 0;
    for (int p = 0; p < ranks; p++) {
      displs[p + 1] = displs[p] + counts[p];
      total += counts[p];
    }
    if (total == 0) {
      break;
    }
    std::vector<real_t> all_pts(static_cast<std::size_t>(total) * dim);
    {
      std::vector<int> ccnt(ranks), cdis(ranks);
      for (int p = 0; p < ranks; p++) {
        ccnt[p] = counts[p] * dim;
        cdis[p] = displs[p] * dim;
      }
      MPI_Allgatherv(my_pts.data(), my_n * dim, MPITypeMap<real_t>::mpi_type,
                     all_pts.data(), ccnt.data(), cdis.data(),
                     MPITypeMap<real_t>::mpi_type, comm);
    }

    DenseMatrix pts(dim, static_cast<int>(total));
    for (long long i = 0; i < total; i++) {
      for (int d = 0; d < dim; d++) {
        pts(d, static_cast<int>(i)) = all_pts[i * dim + d];
      }
    }
    Array<int> elem;
    Array<IntegrationPoint> ips;
    solid_sub->Mesh::FindPoints(pts, elem, ips, false);

    std::vector<int> r_meta;
    std::vector<HYPRE_BigInt> r_cols;
    std::vector<real_t> r_vals;
    {
      Array<int> dofs;
      Vector shape;
      for (long long i = 0; i < total; i++) {
        if (elem[static_cast<int>(i)] < 0) {
          continue;
        }
        const int el = elem[static_cast<int>(i)];
        const auto* fe = solid_fes.GetFE(el);
        shape.SetSize(fe->GetDof());
        fe->CalcShape(ips[static_cast<int>(i)], shape);
        solid_fes.GetElementDofs(el, dofs);
        r_meta.push_back(static_cast<int>(i));
        r_meta.push_back(dofs.Size());
        for (int a = 0; a < dofs.Size(); a++) {
          r_vals.push_back(shape(a));
          for (int k = 0; k < vdim; k++) {
            r_cols.push_back(solid_fes.GetGlobalTDofNumber(
                solid_fes.DofToVDof(dofs[a], k)));
          }
        }
      }
    }
    auto allgather_var = [&](auto& mine, auto mpi_type, auto& all) {
      int n = static_cast<int>(mine.size());
      std::vector<int> cnt(ranks), dis(ranks + 1, 0);
      MPI_Allgather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, comm);
      for (int p = 0; p < ranks; p++) {
        dis[p + 1] = dis[p] + cnt[p];
      }
      all.resize(dis[ranks]);
      MPI_Allgatherv(mine.data(), n, mpi_type, all.data(), cnt.data(),
                     dis.data(), mpi_type, comm);
    };
    std::vector<int> all_meta;
    std::vector<HYPRE_BigInt> all_cols;
    std::vector<real_t> all_vals;
    allgather_var(r_meta, MPI_INT, all_meta);
    allgather_var(r_cols,
                  sizeof(HYPRE_BigInt) == sizeof(long long) ? MPI_LONG_LONG
                                                            : MPI_INT,
                  all_cols);
    allgather_var(r_vals, MPITypeMap<real_t>::mpi_type, all_vals);

    std::size_t cpos = 0, vpos = 0;
    for (std::size_t m = 0; m + 1 < all_meta.size(); m += 2) {
      const int gpt = all_meta[m];
      const int nsh = all_meta[m + 1];
      const std::size_t c0 = cpos, v0 = vpos;
      cpos += static_cast<std::size_t>(nsh) * vdim;
      vpos += nsh;
      if (gpt < displs[rank] || gpt >= displs[rank + 1]) {
        continue;
      }
      const int q = my_qid[gpt - displs[rank]];
      if (resolved[q]) {
        continue;  // first reply wins
      }
      resolved[q] = 1;
      row_cols[q].assign(
          all_cols.begin() + c0,
          all_cols.begin() + c0 + static_cast<std::size_t>(nsh) * vdim);
      row_vals[q].assign(all_vals.begin() + v0, all_vals.begin() + v0 + nsh);
    }
  }
  for (std::size_t q = 0; q < queries.size(); q++) {
    MFEM_VERIFY(resolved[q],
                "NewRadialFluidExtension: an interface projection was not "
                "found on any rank.");
  }

  // Assemble the interior rows (paired rows stay empty; J is added).
  Array<int> I(nrows + 1);
  I = 0;
  std::vector<int> row_query(nrows, -1);
  std::vector<int> row_comp(nrows, 0);
  for (std::size_t q = 0; q < queries.size(); q++) {
    const int sb = queries[q].sdof;
    for (int k = 0; k < vdim; k++) {
      const int lt = fluid_fes.GetLocalTDofNumber(fluid_fes.DofToVDof(sb, k));
      MFEM_VERIFY(lt >= 0, "component ownership mismatch");
      row_query[lt] = static_cast<int>(q);
      row_comp[lt] = k;
      I[lt + 1] = static_cast<int>(row_vals[q].size());
    }
  }
  for (int i = 0; i < nrows; i++) {
    I[i + 1] += I[i];
  }
  const int nnz = I[nrows];
  Array<HYPRE_BigInt> Jc(std::max(nnz, 1));
  Vector data(std::max(nnz, 1));
  for (int i = 0; i < nrows; i++) {
    const int q = row_query[i];
    if (q < 0) {
      continue;
    }
    const int k = row_comp[i];
    const int nsh = static_cast<int>(row_vals[q].size());
    for (int a = 0; a < nsh; a++) {
      Jc[I[i] + a] = row_cols[q][static_cast<std::size_t>(a) * vdim + k];
      data[I[i] + a] = queries[q].t * row_vals[q][a];
    }
  }
  HypreParMatrix E_int(comm, nrows, fluid_fes.GlobalTrueVSize(),
                       solid_fes.GlobalTrueVSize(), I.GetData(), Jc.GetData(),
                       data.GetData(), fluid_fes.GetTrueDofOffsets(),
                       solid_fes.GetTrueDofOffsets());
  return std::unique_ptr<HypreParMatrix>(ParAdd(J.get(), &E_int));
}


ParSlipInterfaceBlocks NewSlipInterfaceMatrix(ParFiniteElementSpace& fes_s,
                                              const HypreParMatrix& J,
                                              const Array<int>& interface_marker,
                                              Coefficient& pi,
                                              Diffeomorphism& map) {
  // The one-sided kernel G on the solid side, on true dofs.
  Array<int> marker(interface_marker);
  ParBilinearForm g(&fes_s);
  g.AddBoundaryIntegrator(new SlipInterfacePressureIntegrator(pi, map),
                          marker);
  g.Assemble();
  g.Finalize();
  std::unique_ptr<HypreParMatrix> G(g.ParallelAssemble());
  std::unique_ptr<HypreParMatrix> Gt(G->Transpose());

  std::unique_ptr<HypreParMatrix> Gsym(
      mfem::Add(0.5, *G, 0.5, *Gt));
  std::unique_ptr<HypreParMatrix> Gskew(
      mfem::Add(0.5, *Gt, -0.5, *G));
  auto* Jnc = const_cast<HypreParMatrix*>(&J);

  // Normal convention as in the serial builder: the minus because the
  // solid-side assembly supplies -N of the derivation.
  ParSlipInterfaceBlocks B;
  B.ss = std::make_unique<HypreParMatrix>(*Gsym);
  *B.ss *= -1.0;
  B.sf.reset(ParMult(Gskew.get(), Jnc));
  *B.sf *= -1.0;
  B.fs.reset(B.sf->Transpose());
  B.ff.reset(mfem::RAP(Gsym.get(), Jnc));
  return B;
}


std::unique_ptr<mfem::HypreParMatrix> NewRadialVacuumExtension(
    ParFiniteElementSpace& body_fes, ParFiniteElementSpace& buffer_fes,
    real_t r_body, real_t r_outer, real_t taper_power, real_t pullback) {
  MFEM_VERIFY(body_fes.FEColl() == buffer_fes.FEColl() &&
                  body_fes.GetVDim() == buffer_fes.GetVDim(),
              "NewRadialVacuumExtension: the spaces must share a "
              "collection and vdim.");
  auto* body_sub = dynamic_cast<ParSubMesh*>(body_fes.GetParMesh());
  auto* buffer_sub = dynamic_cast<ParSubMesh*>(buffer_fes.GetParMesh());
  MFEM_VERIFY(body_sub && buffer_sub &&
                  body_sub->GetParent() == buffer_sub->GetParent(),
              "NewRadialVacuumExtension: both spaces must live on "
              "ParSubMeshes of one parent.");
  MPI_Comm comm = body_fes.GetComm();
  const int dim = body_sub->Dimension();
  const int vdim = body_fes.GetVDim();

  ParFiniteElementSpace parent_fes(
      const_cast<ParMesh*>(static_cast<const ParMesh*>(body_sub->GetParent())),
      const_cast<FiniteElementCollection*>(body_fes.FEColl()), vdim,
      body_fes.GetOrdering());
  SubMeshDofInjection inj_body(body_fes, parent_fes);
  SubMeshDofInjection inj_buffer(buffer_fes, parent_fes);
  auto J = NewSubMeshPairingTrueDofMatrix(inj_buffer, inj_body);

  // Paired (trace) rows of the owned buffer true dofs.
  const int nrows = buffer_fes.GetTrueVSize();
  std::vector<char> paired(nrows, 0);
  {
    SparseMatrix diag, offd;
    HYPRE_BigInt* cmap = nullptr;
    J->GetDiag(diag);
    J->GetOffd(offd, cmap);
    for (int t = 0; t < nrows; t++) {
      if (diag.RowSize(t) + offd.RowSize(t) > 0) {
        paired[t] = 1;
      }
    }
  }

  // Scalar nodal coordinates of the buffer space (local).
  const int ns_buf = buffer_fes.GetNDofs();
  DenseMatrix coords(dim, ns_buf);
  {
    Array<int> dofs;
    Vector x(dim);
    for (int e = 0; e < buffer_sub->GetNE(); e++) {
      const auto* fe = buffer_fes.GetFE(e);
      auto* T = buffer_sub->GetElementTransformation(e);
      buffer_fes.GetElementDofs(e, dofs);
      const auto& nodes = fe->GetNodes();
      for (int i = 0; i < dofs.Size(); i++) {
        T->Transform(nodes.IntPoint(i), x);
        for (int d = 0; d < dim; d++) {
          coords(d, dofs[i]) = x(d);
        }
      }
    }
  }

  // The owned, unpaired scalar nodes with a live taper: the queries.
  struct Query {
    int sdof;
    real_t t, r;
  };
  std::vector<Query> queries;
  for (int sb = 0; sb < ns_buf; sb++) {
    const int lt0 = buffer_fes.GetLocalTDofNumber(buffer_fes.DofToVDof(sb, 0));
    if (lt0 < 0 || paired[lt0]) {
      continue;
    }
    real_t r = 0.0;
    for (int d = 0; d < dim; d++) {
      r += coords(d, sb) * coords(d, sb);
    }
    r = std::sqrt(r);
    real_t t = (r_outer - r) / (r_outer - r_body);
    t = std::min(real_t(1), std::max(real_t(0), t));
    t = std::pow(t, taper_power);
    if (t == 0.0) {
      continue;
    }
    queries.push_back({sb, t, r});
  }

  // Interpolation rows, resolved over retry rounds at shrinking radii.
  const int nsh_max = body_fes.GetTypicalFE()
                          ? body_fes.GetTypicalFE()->GetDof()
                          : 64;
  std::vector<std::vector<HYPRE_BigInt>> row_cols(queries.size());
  std::vector<std::vector<real_t>> row_vals(queries.size());
  std::vector<char> resolved(queries.size(), 0);
  int ranks = 0, rank = 0;
  MPI_Comm_size(comm, &ranks);
  MPI_Comm_rank(comm, &rank);

  for (real_t factor : {1.0, 0.99, 0.97, 0.9}) {
    // Gather the unresolved queries from every rank.
    std::vector<real_t> my_pts;
    std::vector<int> my_qid;
    for (std::size_t q = 0; q < queries.size(); q++) {
      if (resolved[q]) {
        continue;
      }
      const int sb = queries[q].sdof;
      const real_t scale = factor * pullback * r_body /
                           std::max(queries[q].r, real_t(1e-30));
      for (int d = 0; d < dim; d++) {
        my_pts.push_back(scale * coords(d, sb));
      }
      my_qid.push_back(static_cast<int>(q));
    }
    int my_n = static_cast<int>(my_qid.size());
    std::vector<int> counts(ranks), displs(ranks + 1, 0);
    MPI_Allgather(&my_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    long long total = 0;
    for (int p = 0; p < ranks; p++) {
      displs[p + 1] = displs[p] + counts[p];
      total += counts[p];
    }
    if (total == 0) {
      break;
    }
    std::vector<real_t> all_pts(static_cast<std::size_t>(total) * dim);
    {
      std::vector<int> ccnt(ranks), cdis(ranks);
      for (int p = 0; p < ranks; p++) {
        ccnt[p] = counts[p] * dim;
        cdis[p] = displs[p] * dim;
      }
      MPI_Allgatherv(my_pts.data(), my_n * dim,
                     MPITypeMap<real_t>::mpi_type, all_pts.data(),
                     ccnt.data(), cdis.data(), MPITypeMap<real_t>::mpi_type,
                     comm);
    }

    // Locate what this rank's body elements contain.
    DenseMatrix pts(dim, static_cast<int>(total));
    for (long long i = 0; i < total; i++) {
      for (int d = 0; d < dim; d++) {
        pts(d, static_cast<int>(i)) = all_pts[i * dim + d];
      }
    }
    Array<int> elem;
    Array<IntegrationPoint> ips;
    body_sub->Mesh::FindPoints(pts, elem, ips, false);

    // Replies: (global point index, nsh) + global columns + values,
    // shared with every rank; the owner keeps the first reply per point.
    std::vector<int> r_meta;
    std::vector<HYPRE_BigInt> r_cols;
    std::vector<real_t> r_vals;
    {
      Array<int> dofs;
      Vector shape;
      for (long long i = 0; i < total; i++) {
        if (elem[static_cast<int>(i)] < 0) {
          continue;
        }
        const int el = elem[static_cast<int>(i)];
        const auto* fe = body_fes.GetFE(el);
        shape.SetSize(fe->GetDof());
        fe->CalcShape(ips[static_cast<int>(i)], shape);
        body_fes.GetElementDofs(el, dofs);
        r_meta.push_back(static_cast<int>(i));
        r_meta.push_back(dofs.Size());
        for (int a = 0; a < dofs.Size(); a++) {
          r_vals.push_back(shape(a));
          for (int k = 0; k < vdim; k++) {
            r_cols.push_back(body_fes.GetGlobalTDofNumber(
                body_fes.DofToVDof(dofs[a], k)));
          }
        }
      }
    }
    auto allgather_var = [&](auto& mine, auto mpi_type, auto& all) {
      int n = static_cast<int>(mine.size());
      std::vector<int> cnt(ranks), dis(ranks + 1, 0);
      MPI_Allgather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, comm);
      for (int p = 0; p < ranks; p++) {
        dis[p + 1] = dis[p] + cnt[p];
      }
      all.resize(dis[ranks]);
      MPI_Allgatherv(mine.data(), n, mpi_type, all.data(), cnt.data(),
                     dis.data(), mpi_type, comm);
    };
    std::vector<int> all_meta;
    std::vector<HYPRE_BigInt> all_cols;
    std::vector<real_t> all_vals;
    allgather_var(r_meta, MPI_INT, all_meta);
    static_assert(sizeof(HYPRE_BigInt) == sizeof(long long) ||
                      sizeof(HYPRE_BigInt) == sizeof(int),
                  "unexpected HYPRE_BigInt");
    allgather_var(r_cols,
                  sizeof(HYPRE_BigInt) == sizeof(long long) ? MPI_LONG_LONG
                                                            : MPI_INT,
                  all_cols);
    allgather_var(r_vals, MPITypeMap<real_t>::mpi_type, all_vals);

    // Walk the replies; keep those for this rank's unresolved queries.
    std::size_t cpos = 0, vpos = 0;
    for (std::size_t m = 0; m + 1 < all_meta.size(); m += 2) {
      const int gpt = all_meta[m];
      const int nsh = all_meta[m + 1];
      const std::size_t c0 = cpos, v0 = vpos;
      cpos += static_cast<std::size_t>(nsh) * vdim;
      vpos += nsh;
      if (gpt < displs[rank] || gpt >= displs[rank + 1]) {
        continue;
      }
      const int q = my_qid[gpt - displs[rank]];
      if (resolved[q]) {
        continue;  // first reply wins
      }
      resolved[q] = 1;
      row_cols[q].assign(all_cols.begin() + c0,
                         all_cols.begin() + c0 +
                             static_cast<std::size_t>(nsh) * vdim);
      row_vals[q].assign(all_vals.begin() + v0, all_vals.begin() + v0 + nsh);
    }
  }
  for (std::size_t q = 0; q < queries.size(); q++) {
    MFEM_VERIFY(resolved[q],
                "NewRadialVacuumExtension: a surface projection was not "
                "found on any rank.");
  }
  (void)nsh_max;

  // Assemble the interior rows (paired rows stay empty; J is added).
  Array<int> I(nrows + 1);
  I = 0;
  std::vector<int> row_query(nrows, -1);
  std::vector<int> row_comp(nrows, 0);
  for (std::size_t q = 0; q < queries.size(); q++) {
    const int sb = queries[q].sdof;
    for (int k = 0; k < vdim; k++) {
      const int lt = buffer_fes.GetLocalTDofNumber(buffer_fes.DofToVDof(sb, k));
      MFEM_VERIFY(lt >= 0, "component ownership mismatch");
      row_query[lt] = static_cast<int>(q);
      row_comp[lt] = k;
      I[lt + 1] = static_cast<int>(row_vals[q].size());
    }
  }
  for (int i = 0; i < nrows; i++) {
    I[i + 1] += I[i];
  }
  const int nnz = I[nrows];
  Array<HYPRE_BigInt> Jc(std::max(nnz, 1));
  Vector data(std::max(nnz, 1));
  for (int i = 0; i < nrows; i++) {
    const int q = row_query[i];
    if (q < 0) {
      continue;
    }
    const int k = row_comp[i];
    const int nsh = static_cast<int>(row_vals[q].size());
    for (int a = 0; a < nsh; a++) {
      Jc[I[i] + a] = row_cols[q][static_cast<std::size_t>(a) * vdim + k];
      data[I[i] + a] = queries[q].t * row_vals[q][a];
    }
  }
  HypreParMatrix E_int(comm, nrows, buffer_fes.GlobalTrueVSize(),
                       body_fes.GlobalTrueVSize(), I.GetData(), Jc.GetData(),
                       data.GetData(), buffer_fes.GetTrueDofOffsets(),
                       body_fes.GetTrueDofOffsets());
  return std::unique_ptr<HypreParMatrix>(ParAdd(J.get(), &E_int));
}
#endif

// ---------------------------------------------------------------------------
// ReferentialElasticRheology

ReferentialElasticRheology::ReferentialElasticRheology(int dim,
                                                       MatrixCoefficient& C,
                                                       MatrixCoefficient& S_e,
                                                       Diffeomorphism& phi_e)
    : dim_(dim), C_(&C), S_(&S_e), map_(&phi_e) {
  MFEM_VERIFY(dim == 2 || dim == 3,
              "ReferentialElasticRheology: dim must be 2 or 3.");
  const int ns = SymmetricTensorBasis::Size(dim);
  MFEM_VERIFY(C.GetHeight() == ns && C.GetWidth() == ns,
              "ReferentialElasticRheology: C must be n_s x n_s.");
  MFEM_VERIFY(S_e.GetHeight() == dim && S_e.GetWidth() == dim,
              "ReferentialElasticRheology: S_e must be d x d.");
}

Coefficient& ReferentialElasticRheology::RelaxationTime(int) const {
  MFEM_ABORT("ReferentialElasticRheology has no branches.");
}

const RelaxationLaw* ReferentialElasticRheology::Law(int) const {
  MFEM_ABORT("ReferentialElasticRheology has no branches.");
}

void ReferentialElasticRheology::BranchModulus(int, ElementTransformation&,
                                               const IntegrationPoint&,
                                               DenseMatrix&) const {
  MFEM_ABORT("ReferentialElasticRheology has no branches.");
}

void ReferentialElasticRheology::UnrelaxedModulus(ElementTransformation& T,
                                                  const IntegrationPoint& ip,
                                                  DenseMatrix& CU) const {
  C_->Eval(CU, T, ip);
}

std::unique_ptr<ElasticStiffness> ReferentialElasticRheology::MakeStiffness()
    const {
  return std::make_unique<ReferentialStiffness>(*C_, *S_, *map_);
}

// ---------------------------------------------------------------------------
// Construction

LinearQuasiStaticReferentialProblem::LinearQuasiStaticReferentialProblem(
    FiniteElementSpace* fes_u, FiniteElementSpace* fes_zeta,
    const ReferentialElasticRheology& rheology, Coefficient& density,
    real_t gravitational_constant, int dtn_degree,
    Coefficient* background_zeta0)
    : LinearQuasiStaticProblemBase(fes_u, rheology),
      dim_(fes_u->GetMesh()->Dimension()),
      fes_zeta_(fes_zeta),
      ref_rheology_(&rheology),
      rho_(&density),
      G_(gravitational_constant),
      four_pi_G_(4.0 * kPi * gravitational_constant),
      dtn_degree_(dtn_degree),
      one_(1.0),
      inv_four_pi_G_(1.0 / (4.0 * kPi * gravitational_constant)),
      shift_coef_(shift_ / (4.0 * kPi * gravitational_constant)),
      K_zeta_(Operator::MFEM_SPARSEMAT),
      K_shift_(Operator::MFEM_SPARSEMAT),
      C_(Operator::MFEM_SPARSEMAT) {
  MFEM_VERIFY(G_ > 0.0,
              "LinearQuasiStaticReferentialProblem: G must be positive.");
  MFEM_VERIFY(fes_zeta_->GetVDim() == 1,
              "LinearQuasiStaticReferentialProblem: the potential space must "
              "be scalar.");
  MFEM_VERIFY(fes_zeta_->GetMesh()->Dimension() == dim_,
              "LinearQuasiStaticReferentialProblem: mesh dimensions differ.");

  ball_wide_ = (fes_u->GetMesh() == fes_zeta_->GetMesh());
#ifdef MFEM_USE_MPI
  pfes_zeta_ = dynamic_cast<ParFiniteElementSpace*>(fes_zeta_);
  MFEM_VERIFY(
      (pfes_ != nullptr) == (pfes_zeta_ != nullptr),
      "LinearQuasiStaticReferentialProblem: the displacement and potential "
      "spaces must both be serial or both be parallel.");
  if (pfes_zeta_) {
    K_zeta_.SetType(Operator::Hypre_ParCSR);
    K_shift_.SetType(Operator::Hypre_ParCSR);
    C_.SetType(Operator::Hypre_ParCSR);
  }
#endif
  if (!ball_wide_) {
#ifdef MFEM_USE_MPI
    if (pfes_zeta_) {
      auto* psub = dynamic_cast<ParSubMesh*>(pfes_->GetParMesh());
      MFEM_VERIFY(psub && psub->GetParent() == pfes_zeta_->GetParMesh(),
                  "LinearQuasiStaticReferentialProblem: the displacement "
                  "space must live on a ParSubMesh of the potential space's "
                  "mesh, or on that mesh itself.");
      shadow_zeta_ = SubMeshDofInjection::MakeShadowSpace(*pfes_zeta_, *psub);
    } else
#endif
    {
      auto* sub = dynamic_cast<SubMesh*>(fes_->GetMesh());
      MFEM_VERIFY(sub && sub->GetParent() == fes_zeta_->GetMesh(),
                  "LinearQuasiStaticReferentialProblem: the displacement "
                  "space must live on a SubMesh of the potential space's "
                  "mesh, or on that mesh itself.");
      shadow_zeta_ = SubMeshDofInjection::MakeShadowSpace(*fes_zeta_, *sub);
    }
    injection_ =
        std::make_unique<SubMeshDofInjection>(*shadow_zeta_, *fes_zeta_);
  }

  // Potential fields.
  zeta0_ = detail::MakeGridFunction(fes_zeta_);
  zeta_ = detail::MakeGridFunction(fes_zeta_);
  *zeta0_ = 0.0;
  *zeta_ = 0.0;
  if (!ball_wide_) {
    zeta0_shadow_ = detail::MakeGridFunction(shadow_zeta_.get());
    zeta_shadow_ = detail::MakeGridFunction(shadow_zeta_.get());
    *zeta0_shadow_ = 0.0;
    *zeta_shadow_ = 0.0;
  }
  grad_zeta0_shadow_ = std::make_unique<GradientGridFunctionCoefficient>(
      ball_wide_ ? zeta0_.get() : zeta0_shadow_.get());
  Zeta_true_.SetSize(fes_zeta_->GetTrueVSize());
  Zeta_true_ = 0.0;

  // The DtN operator on the outer boundary: the mapping is the identity
  // there, so the plain closure applies (doc/gravitating_elasticity.md §3).
#ifdef MFEM_USE_MPI
  if (pfes_zeta_) {
    dtn_ = std::make_unique<PoissonDtNOperator>(pfes_zeta_->GetComm(),
                                                pfes_zeta_, dtn_degree_);
    dtn_->Assemble();
    dtn_rap_ = std::make_unique<RAPOperator>(dtn_->RAP());
    dtn_op_ = dtn_rap_.get();
  } else
#endif
  {
    dtn_ = std::make_unique<PoissonDtNOperator>(fes_zeta_, dtn_degree_);
    dtn_->Assemble();
    dtn_op_ = dtn_.get();
  }

  // 2-D compatibility data, as in the Eulerian class.
  if (dim_ == 2) {
    auto ones = detail::MakeGridFunction(fes_zeta_);
    *ones = 1.0;
    ones->GetTrueDofs(ones_);
    auto marker = ExternalBoundaryMarker(fes_zeta_->GetMesh());
    auto l = detail::MakeLinearForm(fes_zeta_);
    l->AddBoundaryIntegrator(new BoundaryLFIntegrator(one_), marker);
    l->Assemble();
    ToTrueDofs(*fes_zeta_, *l, L_outer_);
    outer_length_ = Dot(L_outer_, ones_);
    MFEM_VERIFY(outer_length_ > 0.0,
                "LinearQuasiStaticReferentialProblem: empty outer boundary.");
  }

  SetupPotentialOperators();
  ComputeBackgroundPotential(background_zeta0);
  SetupCoupling();
  SetupGravityIntegrators();
  SetupRigidModes();

  b_zeta_ = detail::MakeLinearForm(ball_wide_ ? fes_zeta_
                                              : shadow_zeta_.get());
  B_zeta_.SetSize(fes_zeta_->GetTrueVSize());
  B_zeta_ = 0.0;
}

bool LinearQuasiStaticReferentialProblem::ParallelPotential() const {
#ifdef MFEM_USE_MPI
  return pfes_zeta_ != nullptr;
#else
  return false;
#endif
}

void LinearQuasiStaticReferentialProblem::ToTrueDofs(
    const FiniteElementSpace& fes, const Vector& L, Vector& T) const {
  const Operator* P = fes.GetProlongationMatrix();
  if (P) {
    T.SetSize(P->Width());
    P->MultTranspose(L, T);
  } else {
    T = L;
  }
}

void LinearQuasiStaticReferentialProblem::MakeCompatible(
    Vector& B_zeta) const {
  if (dim_ != 2) {
    return;
  }
  const real_t mass = Dot(B_zeta, ones_);
  B_zeta.Add(-mass / outer_length_, L_outer_);
}

void LinearQuasiStaticReferentialProblem::SetupPotentialOperators() {
  Array<int> empty;
  auto& map = ref_rheology_->EquilibriumMapping();

  // K_a = <a_e grad zeta, grad chi> on the ball (the mapped Laplacian);
  // A_zeta = (K_a + DtN) / 4 pi G.
  k_zeta_form_ = detail::MakeBilinearForm(fes_zeta_);
  k_zeta_form_->AddDomainIntegrator(new TransformedDiffusionIntegrator(map));
  k_zeta_form_->Assemble();
  k_zeta_form_->FormSystemMatrix(empty, K_zeta_);
  const real_t c = 1.0 / four_pi_G_;
  A_zeta_op_ = std::make_unique<SumOperator>(K_zeta_.Ptr(), c, dtn_op_, c,
                                             false, false);
  A_zeta_ = A_zeta_op_.get();

  // Preconditioner: the shifted mapped Laplacian (K_a + eps M) / 4 pi G.
  shift_coef_.constant = shift_ / four_pi_G_;
  k_shift_form_ = detail::MakeBilinearForm(fes_zeta_);
  auto* tdi = new TransformedDiffusionIntegrator(map);
  k_shift_form_->AddDomainIntegrator(tdi);
  k_shift_form_->AddDomainIntegrator(new MassIntegrator(shift_coef_));
  k_shift_form_->Assemble();
  k_shift_form_->FormSystemMatrix(empty, K_shift_);
  {
    // Scale K_shift by 1/4piG through the operator: assemble unscaled and
    // wrap; AMG wants the matrix, so scale the matrix itself instead.
#ifdef MFEM_USE_MPI
    if (pfes_zeta_) {
      *K_shift_.As<HypreParMatrix>() *= c;
      auto amg =
          std::make_unique<HypreBoomerAMG>(*K_shift_.As<HypreParMatrix>());
      amg->SetPrintLevel(0);
      prec_zeta_ = std::move(amg);
    } else
#endif
    {
      *K_shift_.As<SparseMatrix>() *= c;
      prec_zeta_ = std::make_unique<GSSmoother>(*K_shift_.As<SparseMatrix>());
    }
  }

  // CG on A_zeta for the background solve; in 2-D the constant is
  // projected from both sides.
#ifdef MFEM_USE_MPI
  if (pfes_zeta_) {
    cg_zeta_ = std::make_unique<CGSolver>(pfes_zeta_->GetComm());
    if (dim_ == 2) {
      projector_c_ = std::make_unique<NullSpaceProjector>(pfes_zeta_->GetComm());
    }
  } else
#endif
  {
    cg_zeta_ = std::make_unique<CGSolver>();
    if (dim_ == 2) {
      projector_c_ = std::make_unique<NullSpaceProjector>();
    }
  }
  if (dim_ == 2) {
    projector_c_->Add(ones_);
    projected_zeta_op_ =
        std::make_unique<ProjectedOperator>(*A_zeta_, *projector_c_);
    cg_zeta_->SetOperator(*projected_zeta_op_);
    projected_prec_zeta_ = std::make_unique<ProjectedSolver>(*projector_c_);
    projected_prec_zeta_->SetSolver(*prec_zeta_);
    cg_zeta_->SetPreconditioner(*projected_prec_zeta_);
  } else {
    cg_zeta_->SetOperator(*A_zeta_);
    cg_zeta_->SetPreconditioner(*prec_zeta_);
  }
  cg_zeta_->SetRelTol(1e-12);
  cg_zeta_->SetAbsTol(0.0);
  cg_zeta_->SetMaxIter(10000);
  cg_zeta_->iterative_mode = false;
  if (dim_ == 2) {
    projected_zeta_ = std::make_unique<ProjectedSolver>(*projector_c_);
    projected_zeta_->SetSolver(*cg_zeta_);
    projected_zeta_->iterative_mode = false;
    zeta_solver_ = projected_zeta_.get();
  } else {
    zeta_solver_ = cg_zeta_.get();
  }
}

void LinearQuasiStaticReferentialProblem::ComputeBackgroundPotential(
    Coefficient* zeta0) {
  if (zeta0) {
    zeta0_->ProjectCoefficient(*zeta0);
  } else {
    // (K_a + DtN) Zeta0 / 4 pi G = -(rho, chi)_B with the referential
    // density integrated on the SubMesh and injected into the ball.
    Vector bL(fes_zeta_->GetVSize()), B;
    if (ball_wide_) {
      auto rho_form = detail::MakeLinearForm(fes_zeta_);
      rho_form->AddDomainIntegrator(new DomainLFIntegrator(*rho_));
      rho_form->Assemble();
      bL = *rho_form;
    } else {
      auto rho_form = detail::MakeLinearForm(shadow_zeta_.get());
      rho_form->AddDomainIntegrator(new DomainLFIntegrator(*rho_));
      rho_form->Assemble();
      injection_->Mult(*rho_form, bL);
    }
    ToTrueDofs(*fes_zeta_, bL, B);
    B *= -1.0;
    MakeCompatible(B);
    Vector Zeta0(B.Size());
    Zeta0 = 0.0;
    zeta_solver_->Mult(B, Zeta0);
    MFEM_VERIFY(cg_zeta_->GetConverged(),
                "LinearQuasiStaticReferentialProblem: the background "
                "potential solve did not converge.");
    zeta0_->SetFromTrueDofs(Zeta0);
  }
  if (!ball_wide_) {
    injection_->MultTranspose(*zeta0_, *zeta0_shadow_);
  }
}

void LinearQuasiStaticReferentialProblem::SetupCoupling() {
  // c(zeta, v) = (1/4piG) int_B <a'(v) g0, grad zeta>: trial zeta on the
  // ball, test v on the body; C^T by transposition.
  Array<int> empty;
  auto& map = ref_rheology_->EquilibriumMapping();
  const real_t c = 1.0 / four_pi_G_;
  if (ball_wide_) {
#ifdef MFEM_USE_MPI
    if (pfes_zeta_) {
      auto form = std::make_unique<ParMixedBilinearForm>(pfes_zeta_, pfes_);
      form->AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
          map, *grad_zeta0_shadow_, c));
      form->Assemble();
      form->Finalize();
      form->FormRectangularSystemMatrix(empty, empty, C_);
      c_form_ = std::move(form);
      Ct_owned_.reset(C_.As<HypreParMatrix>()->Transpose());
    } else
#endif
    {
      auto form = std::make_unique<MixedBilinearForm>(fes_zeta_, fes_);
      form->AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
          map, *grad_zeta0_shadow_, c));
      form->Assemble();
      form->Finalize();
      form->FormRectangularSystemMatrix(empty, empty, C_);
      c_form_ = std::move(form);
      Ct_owned_.reset(Transpose(*C_.As<SparseMatrix>()));
    }
    C_op_ = C_.Ptr();
    Ct_op_ = Ct_owned_.get();
    return;
  }
#ifdef MFEM_USE_MPI
  if (pfes_zeta_) {
    auto form = std::make_unique<ParSubMeshMixedBilinearForm>(pfes_zeta_, pfes_);
    form->AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
        map, *grad_zeta0_shadow_, c));
    form->Assemble();
    form->FormRectangularSystemMatrix(empty, empty, C_);
    c_form_ = std::move(form);
    Ct_owned_.reset(C_.As<HypreParMatrix>()->Transpose());
  } else
#endif
  {
    auto form = std::make_unique<SubMeshMixedBilinearForm>(fes_zeta_, fes_);
    form->AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
        map, *grad_zeta0_shadow_, c));
    form->Assemble();
    form->FormRectangularSystemMatrix(empty, empty, C_);
    c_form_ = std::move(form);
    Ct_owned_.reset(Transpose(*C_.As<SparseMatrix>()));
  }
  C_op_ = C_.Ptr();
  Ct_op_ = Ct_owned_.get();
}

void LinearQuasiStaticReferentialProblem::SetupGravityIntegrators() {
  auto& map = ref_rheology_->EquilibriumMapping();
  StiffnessIntegrators().AddDomainIntegrator(new ReferentialGravityIntegrator(
      map, *grad_zeta0_shadow_, 1.0 / (2.0 * four_pi_G_)));
}

void LinearQuasiStaticReferentialProblem::SetupRigidModes() {
#ifdef MFEM_USE_MPI
  if (pfes_) {
    projector_u_ = std::make_unique<NullSpaceProjector>(pfes_->GetComm());
  } else
#endif
  {
    projector_u_ = std::make_unique<NullSpaceProjector>();
  }
  auto gf = detail::MakeGridFunction(fes_);
  Vector t;
  auto add = [&](VectorCoefficient& c) {
    gf->ProjectCoefficient(c);
    gf->GetTrueDofs(t);
    projector_u_->Add(t);
  };
  for (int c = 0; c < dim_; c++) {
    Vector e(dim_);
    e = 0.0;
    e[c] = 1.0;
    VectorConstantCoefficient tc(e);
    add(tc);
  }
  auto& map = ref_rheology_->EquilibriumMapping();
  if (dim_ == 2) {
    MappedRotation rot(map, 2);
    add(rot);
  } else {
    for (int c = 0; c < 3; c++) {
      MappedRotation rot(map, c);
      add(rot);
    }
  }

  // The block projector: displacement modes with zero potential partners
  // (doc/gravitating_elasticity.md §3.1), plus the constant in 2-D.
#ifdef MFEM_USE_MPI
  if (pfes_) {
    projector_block_ = std::make_unique<NullSpaceProjector>(pfes_->GetComm());
  } else
#endif
  {
    projector_block_ = std::make_unique<NullSpaceProjector>();
  }
  offsets_.SetSize(3);
  offsets_[0] = 0;
  offsets_[1] = fes_->GetTrueVSize();
  offsets_[2] = fes_zeta_->GetTrueVSize();
  offsets_.PartialSum();
  BlockVector n(offsets_);
  for (int i = 0; i < projector_u_->Size(); i++) {
    n.GetBlock(0) = projector_u_->Basis(i);
    n.GetBlock(1) = 0.0;
    projector_block_->Add(n);
  }
  if (dim_ == 2) {
    n.GetBlock(0) = 0.0;
    n.GetBlock(1) = ones_;
    projector_block_->Add(n);
  }
}

// ---------------------------------------------------------------------------
// Loads

void LinearQuasiStaticReferentialProblem::SetSurfaceLoad(
    Coefficient& sigma, const Array<int>& bdr_marker) {
  MFEM_VERIFY(bdr_marker.Size() == fes_->GetMesh()->bdr_attributes.Max(),
              "SetSurfaceLoad: the marker must be sized to the SubMesh's "
              "bdr_attributes.Max().");
  RegisterTimeDependent(sigma);
  load_markers_.push_back(bdr_marker);
  auto& marker = load_markers_.back();
  auto minus_sigma = std::make_unique<ProductCoefficient>(-1.0, sigma);
  // Potential row only: the u-row load of the Eulerian form is absorbed
  // by the change of variables (doc/gravitating_elasticity.md §3.1).
  b_zeta_->AddBoundaryIntegrator(new BoundaryLFIntegrator(*minus_sigma),
                                 marker);
  load_coefs_.push_back(std::move(minus_sigma));
}


void LinearQuasiStaticReferentialProblem::SetPrescribedVacuumExtension(
    FiniteElementSpace& fes_buffer, const SparseMatrix& E) {
  MFEM_VERIFY(!ball_wide_,
              "SetPrescribedVacuumExtension: for the SubMesh mode (the "
              "ball-wide mode carries its own extension field).");
  MFEM_VERIFY(!ParallelPotential(),
              "SetPrescribedVacuumExtension: serial only at present.");
  MFEM_VERIFY(E.Height() == fes_buffer.GetVSize() &&
                  E.Width() == fes_->GetVSize(),
              "SetPrescribedVacuumExtension: E must map body vdofs to "
              "buffer vdofs.");
  MFEM_VERIFY(!ext_EtGE_, "SetPrescribedVacuumExtension: already set.");

  auto& map = ref_rheology_->EquilibriumMapping();
  auto* buffer_sub = dynamic_cast<SubMesh*>(fes_buffer.GetMesh());
  MFEM_VERIFY(buffer_sub && buffer_sub->GetParent() == fes_zeta_->GetMesh(),
              "SetPrescribedVacuumExtension: the buffer space must live on "
              "a SubMesh of the ball.");

  // zeta0 and its gradient on the buffer.
  shadow_zeta_buffer_ =
      SubMeshDofInjection::MakeShadowSpace(*fes_zeta_, *buffer_sub);
  SubMeshDofInjection inj(*shadow_zeta_buffer_, *fes_zeta_);
  zeta0_buffer_ = detail::MakeGridFunction(shadow_zeta_buffer_.get());
  inj.MultTranspose(*zeta0_, *zeta0_buffer_);
  grad_zeta0_buffer_ =
      std::make_unique<GradientGridFunctionCoefficient>(zeta0_buffer_.get());

  // The buffer's gravity-gravity block, folded: E^T G_V E.
  BilinearForm g_form(&fes_buffer);
  g_form.AddDomainIntegrator(new ReferentialGravityIntegrator(
      map, *grad_zeta0_buffer_, 1.0 / (2.0 * four_pi_G_)));
  g_form.Assemble();
  g_form.Finalize();
  std::unique_ptr<SparseMatrix> Et(Transpose(E));
  std::unique_ptr<SparseMatrix> GE(mfem::Mult(g_form.SpMat(), E));
  ext_EtGE_.reset(mfem::Mult(*Et, *GE));

  // The buffer's coupling, folded onto the body rows: C_total = C + E^T C_V.
  SubMeshMixedBilinearForm c_form(fes_zeta_, &fes_buffer);
  c_form.AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
      map, *grad_zeta0_buffer_, 1.0 / four_pi_G_));
  c_form.Assemble();
  std::unique_ptr<SparseMatrix> EtCv(mfem::Mult(*Et, c_form.SpMat()));
  ext_C_total_.reset(Add(*C_.As<SparseMatrix>(), *EtCv));
  ext_Ct_total_.reset(Transpose(*ext_C_total_));
  C_op_ = ext_C_total_.get();
  Ct_op_ = ext_Ct_total_.get();
  operator_dirty_ = true;
}


#ifdef MFEM_USE_MPI
void LinearQuasiStaticReferentialProblem::SetPrescribedVacuumExtension(
    ParFiniteElementSpace& fes_buffer, const HypreParMatrix& E) {
  MFEM_VERIFY(!ball_wide_ && ParallelPotential(),
              "SetPrescribedVacuumExtension(par): parallel SubMesh mode "
              "only.");
  MFEM_VERIFY(!pext_EtGE_, "SetPrescribedVacuumExtension: already set.");
  auto& map = ref_rheology_->EquilibriumMapping();
  auto* buffer_sub = dynamic_cast<ParSubMesh*>(fes_buffer.GetParMesh());
  MFEM_VERIFY(buffer_sub &&
                  buffer_sub->GetParent() == pfes_zeta_->GetParMesh(),
              "SetPrescribedVacuumExtension: the buffer space must live on "
              "a ParSubMesh of the ball.");

  auto shadow = SubMeshDofInjection::MakeShadowSpace(*pfes_zeta_, *buffer_sub);
  shadow_zeta_buffer_ = std::move(shadow);
  SubMeshDofInjection inj(*shadow_zeta_buffer_, *pfes_zeta_);
  zeta0_buffer_ = detail::MakeGridFunction(shadow_zeta_buffer_.get());
  inj.MultTranspose(*zeta0_, *zeta0_buffer_);
  grad_zeta0_buffer_ =
      std::make_unique<GradientGridFunctionCoefficient>(zeta0_buffer_.get());

  ParBilinearForm g_form(&fes_buffer);
  g_form.AddDomainIntegrator(new ReferentialGravityIntegrator(
      map, *grad_zeta0_buffer_, 1.0 / (2.0 * four_pi_G_)));
  g_form.Assemble();
  g_form.Finalize();
  OperatorHandle Gv(Operator::Hypre_ParCSR);
  Array<int> empty;
  g_form.FormSystemMatrix(empty, Gv);
  pext_EtGE_.reset(
      mfem::RAP(Gv.As<HypreParMatrix>(), const_cast<HypreParMatrix*>(&E)));

  ParSubMeshMixedBilinearForm c_form(pfes_zeta_, &fes_buffer);
  c_form.AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
      map, *grad_zeta0_buffer_, 1.0 / four_pi_G_));
  c_form.Assemble();
  OperatorHandle Cv(Operator::Hypre_ParCSR);
  c_form.FormRectangularSystemMatrix(empty, empty, Cv);
  std::unique_ptr<HypreParMatrix> Et(
      const_cast<HypreParMatrix&>(E).Transpose());
  std::unique_ptr<HypreParMatrix> EtCv(
      ParMult(Et.get(), Cv.As<HypreParMatrix>()));
  pext_C_total_.reset(ParAdd(C_.As<HypreParMatrix>(), EtCv.get()));
  pext_Ct_total_.reset(pext_C_total_->Transpose());
  C_op_ = pext_C_total_.get();
  Ct_op_ = pext_Ct_total_.get();
  operator_dirty_ = true;
}
#endif

void LinearQuasiStaticReferentialProblem::SetVacuumExtension(
    const Array<int>& buffer_marker, Coefficient& mu_gauge, real_t epsilon,
    int refinements) {
  MFEM_VERIFY(ball_wide_,
              "SetVacuumExtension: only for a ball-wide displacement.");
  SetGaugedFluid(buffer_marker, mu_gauge, epsilon, refinements,
                 GaugePenalty::Harmonic);
}

bool LinearQuasiStaticReferentialProblem::GaugeRefine(Vector& X) {
  // As for the gauged fluid's coupled refinement: the physical residual
  // after an exact regularised solve is [eps Q delta_u; 0].
  gauge_residuals_.clear();
  Vector B_zeta_saved(B_zeta_);
  B_zeta_ = 0.0;
  Vector Zeta_acc(Zeta_true_);
  Vector r(X.Size()), d(X.Size()), prev;
  bool ok = true;
  int outer = outer_its_;
  for (int k = 0; k < gauge_refinements_; ++k) {
    Q_.Ptr()->Mult(k == 0 ? X : prev, r);
    gauge_residuals_.push_back(std::sqrt(Dot(r, r)));
    d = 0.0;
    if (X_block_) {
      *X_block_ = 0.0;
    }
    ok = SolveLinearSystem(r, d) && ok;
    outer += outer_its_;
    X += d;
    Zeta_acc += Zeta_true_;
    prev = d;
  }
  B_zeta_ = B_zeta_saved;
  Zeta_true_ = Zeta_acc;
  outer_its_ = outer;
  if (X_block_) {
    X_block_->GetBlock(0) = X;
    X_block_->GetBlock(1) = Zeta_true_;
  }
  DistributePotential(Zeta_true_);
  return ok;
}

void LinearQuasiStaticReferentialProblem::AssembleForce(real_t t) {
  LinearQuasiStaticProblemBase::AssembleForce(t);
  b_zeta_->Assemble();
  if (ball_wide_) {
    ToTrueDofs(*fes_zeta_, *b_zeta_, B_zeta_);
  } else {
    Vector bL(fes_zeta_->GetVSize());
    injection_->Mult(*b_zeta_, bL);
    ToTrueDofs(*fes_zeta_, bL, B_zeta_);
  }
  MakeCompatible(B_zeta_);
}

void LinearQuasiStaticReferentialProblem::RegisterFields(DataCollection& dc) {
  LinearQuasiStaticProblemBase::RegisterFields(dc);
  dc.RegisterField("potential",
                   ball_wide_ ? zeta_.get() : zeta_shadow_.get());
  dc.RegisterField("background_potential",
                   ball_wide_ ? zeta0_.get() : zeta0_shadow_.get());
}

// ---------------------------------------------------------------------------
// Solver

void LinearQuasiStaticReferentialProblem::SetupSolver(OperatorHandle& A) {
  Operator* A_uu = A.Ptr();
  if (ext_EtGE_) {
    A_aug_.Clear();
    A_aug_.Reset(Add(*A.As<SparseMatrix>(), *ext_EtGE_), true);
    A_uu = A_aug_.Ptr();
    SetupDefaultPreconditioner(A_aug_);
#ifdef MFEM_USE_MPI
  } else if (pext_EtGE_) {
    A_aug_.Clear();
    A_aug_.Reset(ParAdd(A.As<HypreParMatrix>(), pext_EtGE_.get()), true);
    A_uu = A_aug_.Ptr();
    SetupDefaultPreconditioner(A_aug_);
#endif
  } else {
    SetupDefaultPreconditioner(A);
  }

  block_op_ = std::make_unique<BlockOperator>(offsets_);
  block_op_->SetBlock(0, 0, A_uu);
  block_op_->SetBlock(0, 1, const_cast<Operator*>(C_op_));
  block_op_->SetBlock(1, 0, const_cast<Operator*>(Ct_op_));
  block_op_->SetBlock(1, 1, const_cast<Operator*>(A_zeta_));

  block_prec_ = std::make_unique<BlockDiagonalPreconditioner>(offsets_);
  block_prec_->SetDiagonalBlock(0, prec_.get());
  block_prec_->SetDiagonalBlock(1, prec_zeta_.get());

#ifdef MFEM_USE_MPI
  if (pfes_) {
    minres_ = std::make_unique<MINRESSolver>(pfes_->GetComm());
  } else
#endif
  {
    minres_ = std::make_unique<MINRESSolver>();
  }
  projected_op_ =
      std::make_unique<ProjectedOperator>(*block_op_, *projector_block_);
  minres_->SetOperator(*projected_op_);
  projected_prec_ = std::make_unique<ProjectedSolver>(*projector_block_);
  projected_prec_->SetSolver(*block_prec_);
  minres_->SetPreconditioner(*projected_prec_);
  minres_->SetRelTol(rel_tol_);
  minres_->SetAbsTol(0.0);
  minres_->SetMaxIter(10000);
  minres_->SetPrintLevel(print_level_);
  minres_->iterative_mode = true;

  projected_ = std::make_unique<ProjectedSolver>(*projector_block_);
  projected_->SetSolver(*minres_);
  projected_->iterative_mode = true;

  if (!X_block_ || X_block_->Size() != offsets_.Last()) {
    X_block_ = std::make_unique<BlockVector>(offsets_);
    *X_block_ = 0.0;
  }
  B_block_ = std::make_unique<BlockVector>(offsets_);
}

bool LinearQuasiStaticReferentialProblem::SolveLinearSystem(const Vector& B,
                                                            Vector& X) {
  B_block_->GetBlock(0) = B;
  B_block_->GetBlock(1) = B_zeta_;
  bool ok = true;
  if (!SetWarmStartTolerance(*minres_, *projected_prec_, *B_block_)) {
    X = 0.0;
    Zeta_true_ = 0.0;
    *X_block_ = 0.0;
  } else {
    projected_->Mult(*B_block_, *X_block_);
    ok = minres_->GetConverged();
    outer_its_ = minres_->GetNumIterations();
    NoteIterations(outer_its_);
    X = X_block_->GetBlock(0);
    projector_u_->Project(X);
    Zeta_true_ = X_block_->GetBlock(1);
  }
  DistributePotential(Zeta_true_);
  return ok;
}

void LinearQuasiStaticReferentialProblem::DistributePotential(
    const Vector& Z) {
  zeta_->SetFromTrueDofs(Z);
  if (!ball_wide_) {
    injection_->MultTranspose(*zeta_, *zeta_shadow_);
  }
}

// ---------------------------------------------------------------------------
// Diagnostics

real_t LinearQuasiStaticReferentialProblem::NullPairResidual(
    const Vector& u_true) {
  EnsureOperator();
  real_t a_max = 0.0;
#ifdef MFEM_USE_MPI
  if (pfes_) {
    auto* hyp = A_.As<HypreParMatrix>();
    SparseMatrix diag, offd;
    HYPRE_BigInt* cmap = nullptr;
    hyp->GetDiag(diag);
    hyp->GetOffd(offd, cmap);
    real_t local = std::max(diag.MaxNorm(), offd.MaxNorm());
    MPI_Allreduce(&local, &a_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                  pfes_->GetComm());
  } else
#endif
  {
    a_max = A_.As<SparseMatrix>()->MaxNorm();
  }
  BlockVector n(offsets_), r(offsets_);
  n.GetBlock(0) = u_true;
  n.GetBlock(1) = 0.0;
  block_op_->Mult(n, r);
  const real_t norm = std::sqrt(Dot(u_true, u_true));
  return std::sqrt(Dot(r, r)) / (a_max * std::max(norm, real_t{1e-300}));
}

std::vector<real_t> LinearQuasiStaticReferentialProblem::RigidPairResiduals() {
  // The projector's basis is orthonormal, so the general diagnostic's
  // norm factor is one and the historic semantics are unchanged.
  EnsureOperator();
  std::vector<real_t> out;
  for (int i = 0; i < projector_u_->Size(); i++) {
    out.push_back(NullPairResidual(projector_u_->Basis(i)));
  }
  return out;
}

// ---------------------------------------------------------------------------
// LinearQuasiStaticSlipReferentialProblem

namespace {

/// rho * sym(D g0_h): the discrete grad-grad-zeta0 matrix coefficient of
/// the mismatch mass term, from the *projected* g0 field so that no
/// density or second potential derivatives are ever taken
/// (doc/slip_interface.tex, discrete realisation). Symmetrised pointwise
/// so the assembled mass matrix is exactly symmetric.
class RhoSymJacobianCoefficient : public MatrixCoefficient {
 public:
  RhoSymJacobianCoefficient(Coefficient& rho, const GridFunction& g0)
      : MatrixCoefficient(g0.FESpace()->GetMesh()->Dimension()),
        rho_(&rho),
        g0_(&g0) {}

  void Eval(DenseMatrix& M, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    T.SetIntPoint(&ip);
    g0_->GetVectorGradient(T, D_);
    const real_t r = rho_->Eval(T, ip);
    M.SetSize(height);
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < height; j++) {
        M(i, j) = 0.5 * r * (D_(i, j) + D_(j, i));
      }
    }
  }

 private:
  Coefficient* rho_;
  const GridFunction* g0_;
  DenseMatrix D_;
};

}  // namespace

LinearQuasiStaticSlipReferentialProblem::
    LinearQuasiStaticSlipReferentialProblem(
        FiniteElementSpace* fes_s, FiniteElementSpace* fes_f,
        FiniteElementSpace* fes_zeta, const ReferentialElasticRheology& rheology,
        Coefficient& density, Coefficient& interface_pressure,
        const Array<int>& interface_marker, real_t gravitational_constant,
        int dtn_degree, Coefficient* background_zeta0)
    : LinearQuasiStaticReferentialProblem(fes_s, fes_zeta, rheology, density,
                                          gravitational_constant, dtn_degree,
                                          background_zeta0),
      fes_f_(fes_f),
      pi_(&interface_pressure),
      interface_marker_(interface_marker) {
  MFEM_VERIFY(!ball_wide_,
              "LinearQuasiStaticSlipReferentialProblem: the solid space "
              "must live on a SubMesh of the ball.");
  MFEM_VERIFY(fes_f_->FEColl() == fes_->FEColl() &&
                  fes_f_->GetVDim() == dim_ &&
                  fes_f_->GetOrdering() == fes_->GetOrdering(),
              "LinearQuasiStaticSlipReferentialProblem: the fluid space "
              "must share the solid space's collection, vdim and ordering.");
  MFEM_VERIFY(
      interface_marker_.Size() == fes_->GetMesh()->bdr_attributes.Max(),
      "LinearQuasiStaticSlipReferentialProblem: the interface marker must "
      "be sized to the solid SubMesh's bdr_attributes.Max().");
#ifdef MFEM_USE_MPI
  pfes_f_ = dynamic_cast<ParFiniteElementSpace*>(fes_f_);
  MFEM_VERIFY((pfes_ != nullptr) == (pfes_f_ != nullptr),
              "LinearQuasiStaticSlipReferentialProblem: the solid and "
              "fluid spaces must both be serial or both be parallel.");
#endif

  u_f_ = detail::MakeGridFunction(fes_f_);
  *u_f_ = 0.0;

  // The interface pairing J = Pi_s^T Pi_f (solid x fluid; vdofs in
  // serial, true dofs in parallel) through a parent vector space, as for
  // the extension builders; and the fluid shadow of the potential space.
#ifdef MFEM_USE_MPI
  if (pfes_f_) {
    auto* fluid_sub = dynamic_cast<ParSubMesh*>(pfes_f_->GetParMesh());
    MFEM_VERIFY(fluid_sub &&
                    fluid_sub->GetParent() == pfes_zeta_->GetParMesh(),
                "LinearQuasiStaticSlipReferentialProblem: the fluid space "
                "must live on a ParSubMesh of the ball.");
    auto* solid_sub = dynamic_cast<ParSubMesh*>(pfes_->GetParMesh());
    ParFiniteElementSpace parent_fes(
        const_cast<ParMesh*>(
            static_cast<const ParMesh*>(solid_sub->GetParent())),
        const_cast<FiniteElementCollection*>(fes_->FEColl()), dim_,
        fes_->GetOrdering());
    SubMeshDofInjection inj_s(*pfes_, parent_fes);
    SubMeshDofInjection inj_f(*pfes_f_, parent_fes);
    pJ_ = NewSubMeshPairingTrueDofMatrix(inj_s, inj_f);
    pJt_.reset(pJ_->Transpose());
    op_J_ = pJ_.get();
    op_Jt_ = pJt_.get();
    shadow_zeta_fluid_ =
        SubMeshDofInjection::MakeShadowSpace(*pfes_zeta_, *fluid_sub);
  } else
#endif
  {
    auto* fluid_sub = dynamic_cast<SubMesh*>(fes_f_->GetMesh());
    MFEM_VERIFY(fluid_sub && fluid_sub->GetParent() == fes_zeta_->GetMesh(),
                "LinearQuasiStaticSlipReferentialProblem: the fluid space "
                "must live on a SubMesh of the ball.");
    auto* solid_sub = dynamic_cast<SubMesh*>(fes_->GetMesh());
    FiniteElementSpace parent_fes(
        const_cast<Mesh*>(static_cast<const Mesh*>(solid_sub->GetParent())),
        const_cast<FiniteElementCollection*>(fes_->FEColl()), dim_,
        fes_->GetOrdering());
    SubMeshDofInjection inj_s(*fes_, parent_fes);
    SubMeshDofInjection inj_f(*fes_f_, parent_fes);
    J_ = NewSubMeshPairingMatrix(inj_s, inj_f);
    Jt_.reset(Transpose(*J_));
    op_J_ = J_.get();
    op_Jt_ = Jt_.get();
    shadow_zeta_fluid_ =
        SubMeshDofInjection::MakeShadowSpace(*fes_zeta_, *fluid_sub);
  }

  // zeta0 and its gradient on the fluid.
  injection_fluid_ =
      std::make_unique<SubMeshDofInjection>(*shadow_zeta_fluid_, *fes_zeta_);
  zeta0_fluid_ = detail::MakeGridFunction(shadow_zeta_fluid_.get());
  grad_zeta0_fluid_ =
      std::make_unique<GradientGridFunctionCoefficient>(zeta0_fluid_.get());

  // The base class solved the background with the solid mass only; redo
  // it with the fluid included (unless zeta0 was prescribed).
  if (!background_zeta0) {
    RecomputeBackgroundPotential();
  }
  injection_fluid_->MultTranspose(*zeta0_, *zeta0_fluid_);

  SetupSlipRigidModes();
}

void LinearQuasiStaticSlipReferentialProblem::RecomputeBackgroundPotential() {
  // As the base class's background solve, with the fluid's referential
  // mass added to the source.
  Vector bL(fes_zeta_->GetVSize()), tmp(fes_zeta_->GetVSize()), B;
  {
    auto rho_form = detail::MakeLinearForm(shadow_zeta_.get());
    rho_form->AddDomainIntegrator(new DomainLFIntegrator(*rho_));
    rho_form->Assemble();
    injection_->Mult(*rho_form, bL);
  }
  {
    auto rho_form = detail::MakeLinearForm(shadow_zeta_fluid_.get());
    rho_form->AddDomainIntegrator(new DomainLFIntegrator(*rho_));
    rho_form->Assemble();
    injection_fluid_->Mult(*rho_form, tmp);
    bL += tmp;
  }
  ToTrueDofs(*fes_zeta_, bL, B);
  B *= -1.0;
  MakeCompatible(B);
  Vector Zeta0(B.Size());
  Zeta0 = 0.0;
  zeta_solver_->Mult(B, Zeta0);
  MFEM_VERIFY(cg_zeta_->GetConverged(),
              "LinearQuasiStaticSlipReferentialProblem: the background "
              "potential solve did not converge.");
  zeta0_->SetFromTrueDofs(Zeta0);
  injection_->MultTranspose(*zeta0_, *zeta0_shadow_);
  // The solid coupling was assembled from the solid-only zeta0: rebuild.
  SetupCoupling();
}

void LinearQuasiStaticSlipReferentialProblem::SetupSlipRigidModes() {
  offsets3_.SetSize(4);
  offsets3_[0] = 0;
  offsets3_[1] = fes_->GetTrueVSize();
  offsets3_[2] = fes_f_->GetTrueVSize();
  offsets3_[3] = fes_zeta_->GetTrueVSize();
  offsets3_.PartialSum();

  // Common translations, independent mapped rotations of shell and core
  // (a frictionless axisymmetric interface transmits no torque), the 2-D
  // potential constant; all with zero potential partners (near-null with
  // the tapered extensions, as for the vacuum extension).
#ifdef MFEM_USE_MPI
  if (pfes_) {
    projector3_ = std::make_unique<NullSpaceProjector>(pfes_->GetComm());
  } else
#endif
  {
    projector3_ = std::make_unique<NullSpaceProjector>();
  }
  auto gf_s = detail::MakeGridFunction(fes_);
  auto gf_f = detail::MakeGridFunction(fes_f_);
  Vector ts, tf;
  BlockVector n(offsets3_);
  for (int c = 0; c < dim_; c++) {
    Vector e(dim_);
    e = 0.0;
    e[c] = 1.0;
    VectorConstantCoefficient tc(e);
    gf_s->ProjectCoefficient(tc);
    gf_s->GetTrueDofs(ts);
    gf_f->ProjectCoefficient(tc);
    gf_f->GetTrueDofs(tf);
    n = 0.0;
    n.GetBlock(0) = ts;
    n.GetBlock(1) = tf;
    projector3_->Add(n);
  }
  auto& map = ref_rheology_->EquilibriumMapping();
  const int nrot = (dim_ == 2) ? 1 : 3;
  for (int c = 0; c < nrot; c++) {
    MappedRotation rot(map, dim_ == 2 ? 2 : c);
    gf_s->ProjectCoefficient(rot);
    gf_s->GetTrueDofs(ts);
    n = 0.0;
    n.GetBlock(0) = ts;
    projector3_->Add(n);
    gf_f->ProjectCoefficient(rot);
    gf_f->GetTrueDofs(tf);
    n = 0.0;
    n.GetBlock(1) = tf;
    projector3_->Add(n);
  }
  if (dim_ == 2) {
    n = 0.0;
    n.GetBlock(2) = ones_;
    projector3_->Add(n);
  }
}

void LinearQuasiStaticSlipReferentialProblem::SetFluidExtension(
    const SparseMatrix& E) {
  MFEM_VERIFY(!ParallelPotential(),
              "SetFluidExtension: the sparse overload is for the serial "
              "problem.");
  MFEM_VERIFY(E.Height() == fes_f_->GetVSize() && E.Width() == fes_->GetVSize(),
              "SetFluidExtension: E must map solid vdofs to fluid vdofs.");
  Ef_ = std::make_unique<SparseMatrix>(E);
  operator_dirty_ = true;
}

#ifdef MFEM_USE_MPI
void LinearQuasiStaticSlipReferentialProblem::SetFluidExtension(
    const HypreParMatrix& E) {
  MFEM_VERIFY(ParallelPotential(),
              "SetFluidExtension: the hypre overload is for the parallel "
              "problem.");
  MFEM_VERIFY(E.Height() == pfes_f_->GetTrueVSize() &&
                  E.Width() == pfes_->GetTrueVSize(),
              "SetFluidExtension: E must map solid true dofs to fluid "
              "true dofs.");
  pEf_ = std::make_unique<HypreParMatrix>(E);
  operator_dirty_ = true;
}
#endif

void LinearQuasiStaticSlipReferentialProblem::SetFluidGauge(
    Coefficient& mu_gauge, real_t epsilon) {
  MFEM_VERIFY(epsilon > 0.0, "SetFluidGauge: epsilon must be positive.");
  fluid_mu_gauge_ = &mu_gauge;
  fluid_gauge_eps_ = epsilon;
  operator_dirty_ = true;
}

void LinearQuasiStaticSlipReferentialProblem::SetConstraint(
    real_t theta, int al_iterations) {
  MFEM_VERIFY(theta > 0.0 && al_iterations >= 1,
              "SetConstraint: theta must be positive and al_iterations at "
              "least one.");
  theta_ = theta;
  al_iterations_ = al_iterations;
  operator_dirty_ = true;
}

void LinearQuasiStaticSlipReferentialProblem::SetGaugedFluid(
    const Array<int>&, Coefficient&, real_t, int, GaugePenalty) {
  MFEM_ABORT(
      "LinearQuasiStaticSlipReferentialProblem: the fluid has its own "
      "space here; use SetFluidGauge().");
}

void LinearQuasiStaticSlipReferentialProblem::AssembleSlipBlocks(
    OperatorHandle& A) {
  MFEM_VERIFY(Ef_,
              "LinearQuasiStaticSlipReferentialProblem: call "
              "SetFluidExtension() before the first Solve().");
  MFEM_VERIFY(fluid_mu_gauge_,
              "LinearQuasiStaticSlipReferentialProblem: call "
              "SetFluidGauge() before the first Solve().");
  auto& map = ref_rheology_->EquilibriumMapping();
  std::unique_ptr<SparseMatrix> Eft(Transpose(*Ef_));

  // Fluid elastic block with the fluid's own field: material + geometric
  // (the dictionary mu_b = pi is the rheology's business).
  BilinearForm a_f(fes_f_);
  {
    auto stiffness = ref_rheology_->MakeStiffness();
    stiffness->AddIntegrators(a_f, nullptr);
  }
  a_f.Assemble();
  a_f.Finalize();

  // Fluid-region a'' gravity with the extension field, folded: E^T G_F E.
  BilinearForm g_f(fes_f_);
  g_f.AddDomainIntegrator(new ReferentialGravityIntegrator(
      map, *grad_zeta0_fluid_, 1.0 / (2.0 * four_pi_G_)));
  g_f.Assemble();
  g_f.Finalize();
  std::unique_ptr<SparseMatrix> GfE(mfem::Mult(g_f.SpMat(), *Ef_));
  std::unique_ptr<SparseMatrix> EtGfE(mfem::Mult(*Eft, *GfE));

  // Fluid-region a' coupling with the extension field: E^T C_F.
  SubMeshMixedBilinearForm c_f(fes_zeta_, fes_f_);
  c_f.AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
      map, *grad_zeta0_fluid_, 1.0 / four_pi_G_));
  c_f.Assemble();
  std::unique_ptr<SparseMatrix> EtCf(mfem::Mult(*Eft, c_f.SpMat()));

  // The mismatch pieces (doc/slip_interface.tex, discrete realisation;
  // phi_e = id for now — the mapped variants are deferred with the
  // mapped discrete-gravity unit). K_c = int_Bf rho w . grad zeta1:
  SubMeshMixedBilinearForm kc(fes_zeta_, fes_f_);
  kc.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(*rho_));
  kc.Assemble();
  const SparseMatrix& Kc = kc.SpMat();
  std::unique_ptr<SparseMatrix> EtKc(mfem::Mult(*Eft, Kc));

  // The projected g0 field, its rho sym(D g0) mass matrix, and the
  // product-rule form G_c = int rho w . grad(g0 . u).
  g0_fluid_gf_ = detail::MakeGridFunction(fes_f_);
  g0_fluid_gf_->ProjectDiscCoefficient(*grad_zeta0_fluid_,
                                       GridFunction::ARITHMETIC);
  RhoSymJacobianCoefficient hess_c(*rho_, *g0_fluid_gf_);
  BilinearForm m_f(fes_f_);
  m_f.AddDomainIntegrator(new VectorMassIntegrator(hess_c));
  m_f.Assemble();
  m_f.Finalize();
  const SparseMatrix& Mt = m_f.SpMat();
  VectorGridFunctionCoefficient g0_c(g0_fluid_gf_.get());
  BilinearForm gc_f(fes_f_);
  gc_f.AddDomainIntegrator(new DomainVectorGradVectorIntegrator(g0_c, *rho_));
  gc_f.Assemble();
  gc_f.Finalize();
  const SparseMatrix& Gc = gc_f.SpMat();

  // Gravity-mismatch folds: the Hessian contribution
  //   w^T Mt w - 2 w^T (Gc - Mt) vtil,   w = u_f - E u_s, vtil = E u_s,
  // collapses to the blocks
  //   (1,1) += Mt,  (1,0) += -Gc E,  (0,0) += E^T (Gc + Gc^T - Mt) E.
  std::unique_ptr<SparseMatrix> GcE(mfem::Mult(Gc, *Ef_));
  std::unique_ptr<SparseMatrix> fold00;
  {
    std::unique_ptr<SparseMatrix> Gct(Transpose(Gc));
    std::unique_ptr<SparseMatrix> S1(Add(1.0, Gc, 1.0, *Gct));
    std::unique_ptr<SparseMatrix> S2(Add(1.0, *S1, -1.0, Mt));
    std::unique_ptr<SparseMatrix> SE(mfem::Mult(*S2, *Ef_));
    fold00.reset(mfem::Mult(*Eft, *SE));
  }

  // The interface pressure form B_Sigma and the constraint penalty
  // pieces theta [Bn, -Bn J; -J^T Bn, J^T Bn J].
  auto B = NewSlipInterfaceMatrix(*fes_, *J_, interface_marker_, *pi_, map);
  {
    Array<int> marker(interface_marker_);
    BilinearForm bn(fes_);
    bn.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(map), marker);
    bn.Assemble();
    bn.Finalize();
    Bn_ = std::make_unique<SparseMatrix>(bn.SpMat());
  }
  std::unique_ptr<SparseMatrix> BnJ(mfem::Mult(*Bn_, *J_));
  std::unique_ptr<SparseMatrix> JtBn(Transpose(*BnJ));
  std::unique_ptr<SparseMatrix> JtBnJ(mfem::Mult(*Jt_, *BnJ));

  // The fluid gauge penalty eps Q (solver operator only).
  {
    ConstantCoefficient eps_c(fluid_gauge_eps_);
    ProductCoefficient mu_eps(eps_c, *fluid_mu_gauge_);
    BilinearForm qf(fes_f_);
    qf.AddDomainIntegrator(new ElasticityIntegrator(mu_eps, -2.0 / dim_, 1.0));
    qf.Assemble();
    qf.Finalize();
    Qf_ = std::make_unique<SparseMatrix>(qf.SpMat());
  }

  // Physical blocks. Solid row: base stiffness (with the vacuum-extension
  // fold, as the base SetupSolver would apply it) + the fluid folds + the
  // interface form.
  {
    std::unique_ptr<SparseMatrix> acc(new SparseMatrix(*A.As<SparseMatrix>()));
    if (ext_EtGE_) {
      acc.reset(Add(1.0, *acc, 1.0, *ext_EtGE_));
    }
    acc.reset(Add(1.0, *acc, 1.0, *EtGfE));
    acc.reset(Add(1.0, *acc, 1.0, *fold00));
    acc.reset(Add(1.0, *acc, 1.0, *B.ss));
    A00_ = std::move(acc);
  }
  {
    std::unique_ptr<SparseMatrix> acc(new SparseMatrix(*B.fs));
    acc.reset(Add(1.0, *acc, -1.0, *GcE));
    A10_ = std::move(acc);
    A01_.reset(Transpose(*A10_));
  }
  {
    std::unique_ptr<SparseMatrix> acc(new SparseMatrix(a_f.SpMat()));
    acc.reset(Add(1.0, *acc, 1.0, Mt));
    acc.reset(Add(1.0, *acc, 1.0, *B.ff));
    A11_ = std::move(acc);
  }
  {
    const SparseMatrix& C_solid =
        ext_C_total_ ? *ext_C_total_ : *C_.As<SparseMatrix>();
    std::unique_ptr<SparseMatrix> acc(new SparseMatrix(C_solid));
    acc.reset(Add(1.0, *acc, 1.0, *EtCf));
    acc.reset(Add(1.0, *acc, -1.0, *EtKc));
    A02_ = std::move(acc);
    A20_.reset(Transpose(*A02_));
  }
  A12_ = std::make_unique<SparseMatrix>(Kc);
  A21_.reset(Transpose(*A12_));

  // Solver blocks: physical + penalty (+ eps Q on the fluid diagonal).
  S00_.reset(Add(1.0, *A00_, theta_, *Bn_));
  S10_.reset(Add(1.0, *A10_, -theta_, *JtBn));
  S01_.reset(Transpose(*S10_));
  {
    std::unique_ptr<SparseMatrix> acc(Add(1.0, *A11_, theta_, *JtBnJ));
    A11_solve_.reset(Add(1.0, *acc, 1.0, *Qf_));
  }

  op_A00_ = A00_.get();
  op_A01_ = A01_.get();
  op_A10_ = A10_.get();
  op_A11_ = A11_.get();
  op_A02_ = A02_.get();
  op_A20_ = A20_.get();
  op_A12_ = A12_.get();
  op_A21_ = A21_.get();
  op_S00_ = S00_.get();
  op_S01_ = S01_.get();
  op_S10_ = S10_.get();
  op_S11_ = A11_solve_.get();
  op_Qf_ = Qf_.get();
  op_Bn_ = Bn_.get();
  prec11_ = std::make_unique<GSSmoother>(*A11_solve_);
}

#ifdef MFEM_USE_MPI
void LinearQuasiStaticSlipReferentialProblem::AssembleSlipBlocksPar(
    OperatorHandle& A) {
  MFEM_VERIFY(pEf_,
              "LinearQuasiStaticSlipReferentialProblem: call "
              "SetFluidExtension() before the first Solve().");
  MFEM_VERIFY(fluid_mu_gauge_,
              "LinearQuasiStaticSlipReferentialProblem: call "
              "SetFluidGauge() before the first Solve().");
  auto& map = ref_rheology_->EquilibriumMapping();
  Array<int> empty;
  std::unique_ptr<HypreParMatrix> Eft(pEf_->Transpose());

  // Fluid elastic block with the fluid's own field.
  OperatorHandle Aff(Operator::Hypre_ParCSR);
  ParBilinearForm a_f(pfes_f_);
  {
    auto stiffness = ref_rheology_->MakeStiffness();
    stiffness->AddIntegrators(a_f, nullptr);
  }
  a_f.Assemble();
  a_f.Finalize();
  a_f.FormSystemMatrix(empty, Aff);

  // Fluid-region a'' gravity with the extension field, folded E^T G_F E.
  std::unique_ptr<HypreParMatrix> EtGfE;
  {
    OperatorHandle Gf(Operator::Hypre_ParCSR);
    ParBilinearForm g_f(pfes_f_);
    g_f.AddDomainIntegrator(new ReferentialGravityIntegrator(
        map, *grad_zeta0_fluid_, 1.0 / (2.0 * four_pi_G_)));
    g_f.Assemble();
    g_f.Finalize();
    g_f.FormSystemMatrix(empty, Gf);
    EtGfE.reset(mfem::RAP(Gf.As<HypreParMatrix>(), pEf_.get()));
  }

  // Fluid-region a' coupling with the extension field: E^T C_F.
  std::unique_ptr<HypreParMatrix> EtCf;
  {
    OperatorHandle Cf(Operator::Hypre_ParCSR);
    ParSubMeshMixedBilinearForm c_f(pfes_zeta_, pfes_f_);
    c_f.AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
        map, *grad_zeta0_fluid_, 1.0 / four_pi_G_));
    c_f.Assemble();
    c_f.FormRectangularSystemMatrix(empty, empty, Cf);
    EtCf.reset(ParMult(Eft.get(), Cf.As<HypreParMatrix>()));
  }

  // Mismatch coupling K_c (phi_e = id, as in serial).
  std::unique_ptr<HypreParMatrix> Kc, EtKc;
  {
    OperatorHandle KcH(Operator::Hypre_ParCSR);
    ParSubMeshMixedBilinearForm kc(pfes_zeta_, pfes_f_);
    kc.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(*rho_));
    kc.Assemble();
    kc.FormRectangularSystemMatrix(empty, empty, KcH);
    Kc = std::make_unique<HypreParMatrix>(*KcH.As<HypreParMatrix>());
    EtKc.reset(ParMult(Eft.get(), Kc.get()));
  }

  // The projected g0 field and the two w-term matrices.
  g0_fluid_gf_ = detail::MakeGridFunction(fes_f_);
  g0_fluid_gf_->ProjectDiscCoefficient(*grad_zeta0_fluid_,
                                       GridFunction::ARITHMETIC);
  OperatorHandle Mt(Operator::Hypre_ParCSR), Gc(Operator::Hypre_ParCSR);
  RhoSymJacobianCoefficient hess_c(*rho_, *g0_fluid_gf_);
  ParBilinearForm m_f(pfes_f_);
  m_f.AddDomainIntegrator(new VectorMassIntegrator(hess_c));
  m_f.Assemble();
  m_f.Finalize();
  m_f.FormSystemMatrix(empty, Mt);
  VectorGridFunctionCoefficient g0_c(g0_fluid_gf_.get());
  ParBilinearForm gc_f(pfes_f_);
  gc_f.AddDomainIntegrator(new DomainVectorGradVectorIntegrator(g0_c, *rho_));
  gc_f.Assemble();
  gc_f.Finalize();
  gc_f.FormSystemMatrix(empty, Gc);

  // Gravity-mismatch folds, as in serial:
  //   (1,1) += Mt, (1,0) += -Gc E, (0,0) += E^T (Gc + Gc^T - Mt) E.
  std::unique_ptr<HypreParMatrix> GcE(
      ParMult(Gc.As<HypreParMatrix>(), pEf_.get()));
  std::unique_ptr<HypreParMatrix> fold00;
  {
    std::unique_ptr<HypreParMatrix> Gct(Gc.As<HypreParMatrix>()->Transpose());
    std::unique_ptr<HypreParMatrix> S1(
        mfem::Add(1.0, *Gc.As<HypreParMatrix>(), 1.0, *Gct));
    std::unique_ptr<HypreParMatrix> S2(
        mfem::Add(1.0, *S1, -1.0, *Mt.As<HypreParMatrix>()));
    std::unique_ptr<HypreParMatrix> EtS2(ParMult(Eft.get(), S2.get()));
    fold00.reset(ParMult(EtS2.get(), pEf_.get()));
  }

  // The interface pressure form and the constraint penalty pieces.
  auto B = NewSlipInterfaceMatrix(*pfes_, *pJ_, interface_marker_, *pi_, map);
  {
    OperatorHandle BnH(Operator::Hypre_ParCSR);
    Array<int> marker(interface_marker_);
    ParBilinearForm bn(pfes_);
    bn.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(map), marker);
    bn.Assemble();
    bn.Finalize();
    bn.FormSystemMatrix(empty, BnH);
    pBn_ = std::make_unique<HypreParMatrix>(*BnH.As<HypreParMatrix>());
  }
  std::unique_ptr<HypreParMatrix> BnJ(ParMult(pBn_.get(), pJ_.get()));
  std::unique_ptr<HypreParMatrix> JtBn(BnJ->Transpose());
  std::unique_ptr<HypreParMatrix> JtBnJ(mfem::RAP(pBn_.get(), pJ_.get()));

  // The fluid gauge penalty eps Q.
  {
    OperatorHandle QfH(Operator::Hypre_ParCSR);
    ConstantCoefficient eps_c(fluid_gauge_eps_);
    ProductCoefficient mu_eps(eps_c, *fluid_mu_gauge_);
    ParBilinearForm qf(pfes_f_);
    qf.AddDomainIntegrator(new ElasticityIntegrator(mu_eps, -2.0 / dim_, 1.0));
    qf.Assemble();
    qf.Finalize();
    qf.FormSystemMatrix(empty, QfH);
    pQf_ = std::make_unique<HypreParMatrix>(*QfH.As<HypreParMatrix>());
  }

  // Physical blocks (solid row with the vacuum-extension fold as the
  // base SetupSolver would apply it), then the solver blocks.
  {
    std::unique_ptr<HypreParMatrix> acc(
        mfem::Add(1.0, *A.As<HypreParMatrix>(), 1.0, *EtGfE));
    if (pext_EtGE_) {
      acc.reset(mfem::Add(1.0, *acc, 1.0, *pext_EtGE_));
    }
    acc.reset(mfem::Add(1.0, *acc, 1.0, *fold00));
    pA00_.reset(mfem::Add(1.0, *acc, 1.0, *B.ss));
  }
  {
    std::unique_ptr<HypreParMatrix> acc(mfem::Add(1.0, *B.fs, -1.0, *GcE));
    pA10_ = std::move(acc);
    pA01_.reset(pA10_->Transpose());
  }
  {
    std::unique_ptr<HypreParMatrix> acc(
        mfem::Add(1.0, *Aff.As<HypreParMatrix>(), 1.0,
                  *Mt.As<HypreParMatrix>()));
    pA11_.reset(mfem::Add(1.0, *acc, 1.0, *B.ff));
  }
  {
    const HypreParMatrix& C_solid =
        pext_C_total_ ? *pext_C_total_ : *C_.As<HypreParMatrix>();
    std::unique_ptr<HypreParMatrix> acc(mfem::Add(1.0, C_solid, 1.0, *EtCf));
    pA02_.reset(mfem::Add(1.0, *acc, -1.0, *EtKc));
    pA20_.reset(pA02_->Transpose());
  }
  pA12_ = std::move(Kc);
  pA21_.reset(pA12_->Transpose());

  pS00_.reset(mfem::Add(1.0, *pA00_, theta_, *pBn_));
  pS10_.reset(mfem::Add(1.0, *pA10_, -theta_, *JtBn));
  pS01_.reset(pS10_->Transpose());
  {
    std::unique_ptr<HypreParMatrix> acc(
        mfem::Add(1.0, *pA11_, theta_, *JtBnJ));
    pA11_solve_.reset(mfem::Add(1.0, *acc, 1.0, *pQf_));
  }

  op_A00_ = pA00_.get();
  op_A01_ = pA01_.get();
  op_A10_ = pA10_.get();
  op_A11_ = pA11_.get();
  op_A02_ = pA02_.get();
  op_A20_ = pA20_.get();
  op_A12_ = pA12_.get();
  op_A21_ = pA21_.get();
  op_S00_ = pS00_.get();
  op_S01_ = pS01_.get();
  op_S10_ = pS10_.get();
  op_S11_ = pA11_solve_.get();
  op_Qf_ = pQf_.get();
  op_Bn_ = pBn_.get();
  {
    auto amg = std::make_unique<HypreBoomerAMG>(*pA11_solve_);
    amg->SetSystemsOptions(dim_);
    amg->SetPrintLevel(0);
    prec11_ = std::move(amg);
  }
}
#endif

void LinearQuasiStaticSlipReferentialProblem::SetupSolver(OperatorHandle& A) {
#ifdef MFEM_USE_MPI
  if (pfes_) {
    AssembleSlipBlocksPar(A);
  } else
#endif
  {
    AssembleSlipBlocks(A);
  }

  block_op3_ = std::make_unique<BlockOperator>(offsets3_);
  block_op3_->SetBlock(0, 0, const_cast<Operator*>(op_S00_));
  block_op3_->SetBlock(0, 1, const_cast<Operator*>(op_S01_));
  block_op3_->SetBlock(1, 0, const_cast<Operator*>(op_S10_));
  block_op3_->SetBlock(1, 1, const_cast<Operator*>(op_S11_));
  block_op3_->SetBlock(0, 2, const_cast<Operator*>(op_A02_));
  block_op3_->SetBlock(2, 0, const_cast<Operator*>(op_A20_));
  block_op3_->SetBlock(1, 2, const_cast<Operator*>(op_A12_));
  block_op3_->SetBlock(2, 1, const_cast<Operator*>(op_A21_));
  block_op3_->SetBlock(2, 2, const_cast<Operator*>(A_zeta_));

  // Block-diagonal preconditioner: the base class's default on the solid
  // solver block (GS serial, elasticity BoomerAMG parallel), GS/AMG on
  // the fluid block (built by the assembly path), the shifted mapped
  // Laplacian on the potential.
  {
    OperatorHandle S00h;
#ifdef MFEM_USE_MPI
    if (pfes_) {
      S00h.Reset(pS00_.get(), false);
    } else
#endif
    {
      S00h.Reset(S00_.get(), false);
    }
    prec_stale_ = true;  // the solver block changes with every assembly
    SetupDefaultPreconditioner(S00h);
  }
  block_prec3_ = std::make_unique<BlockDiagonalPreconditioner>(offsets3_);
  block_prec3_->SetDiagonalBlock(0, prec_.get());
  block_prec3_->SetDiagonalBlock(1, prec11_.get());
  block_prec3_->SetDiagonalBlock(2, prec_zeta_.get());

#ifdef MFEM_USE_MPI
  if (pfes_) {
    minres3_ = std::make_unique<MINRESSolver>(pfes_->GetComm());
  } else
#endif
  {
    minres3_ = std::make_unique<MINRESSolver>();
  }
  projected_op3_ =
      std::make_unique<ProjectedOperator>(*block_op3_, *projector3_);
  minres3_->SetOperator(*projected_op3_);
  projected_prec3_ = std::make_unique<ProjectedSolver>(*projector3_);
  projected_prec3_->SetSolver(*block_prec3_);
  minres3_->SetPreconditioner(*projected_prec3_);
  minres3_->SetRelTol(rel_tol_);
  minres3_->SetAbsTol(0.0);
  minres3_->SetMaxIter(10000);
  minres3_->SetPrintLevel(print_level_);
  minres3_->iterative_mode = true;

  projected3_ = std::make_unique<ProjectedSolver>(*projector3_);
  projected3_->SetSolver(*minres3_);
  projected3_->iterative_mode = true;

  if (!X3_ || X3_->Size() != offsets3_.Last()) {
    X3_ = std::make_unique<BlockVector>(offsets3_);
    *X3_ = 0.0;
  }
  B3_ = std::make_unique<BlockVector>(offsets3_);
  w_al_ = std::make_unique<BlockVector>(offsets3_);
  *w_al_ = 0.0;
}

bool LinearQuasiStaticSlipReferentialProblem::SolveLinearSystem(
    const Vector& B, Vector& X) {
  B3_->GetBlock(0) = B;
  B3_->GetBlock(1) = 0.0;
  B3_->GetBlock(2) = B_zeta_;
  jump_history_.clear();
  *w_al_ = 0.0;
  if (std::sqrt(Dot(*B3_, *B3_)) == 0.0) {
    *X3_ = 0.0;
    X = 0.0;
    Zeta_true_ = 0.0;
    *u_f_ = 0.0;
    DistributePotential(Zeta_true_);
    return true;
  }

  // Augmented-Lagrangian iterations for the normal-jump constraint,
  // interleaved with the Tikhonov refinement of the fluid gauge (the
  // sliding-interface scheme of doc/gauge_penalty_iteration.tex §4):
  //   S U_{k+1} = F - w_k + eps Q u_{f,k},   w_{k+1} = w_k + theta P U.
  bool ok = true;
  int outer = 0;
  BlockVector rhs(offsets3_);
  Vector js(offsets3_[1]), tmp_s(offsets3_[1]), Bjs(offsets3_[1]);
  Vector tmp_f(fes_f_->GetTrueVSize());
  for (int k = 0; k < al_iterations_; k++) {
    rhs = *B3_;
    rhs -= *w_al_;
    op_Qf_->AddMult(X3_->GetBlock(1), rhs.GetBlock(1));
    projected3_->Mult(rhs, *X3_);
    ok = minres3_->GetConverged() && ok;
    outer += minres3_->GetNumIterations();

    // js = u_s - J u_f; w += theta [Bn js; -J^T Bn js].
    js = X3_->GetBlock(0);
    op_J_->Mult(X3_->GetBlock(1), tmp_s);
    js -= tmp_s;
    op_Bn_->Mult(js, Bjs);
    jump_history_.push_back(std::sqrt(std::abs(Dot(js, Bjs))));
    w_al_->GetBlock(0).Add(theta_, Bjs);
    op_Jt_->Mult(Bjs, tmp_f);
    w_al_->GetBlock(1).Add(-theta_, tmp_f);
  }
  outer_its_ = outer;
  NoteIterations(outer);

  X = X3_->GetBlock(0);
  Zeta_true_ = X3_->GetBlock(2);
  u_f_->SetFromTrueDofs(X3_->GetBlock(1));
  DistributePotential(Zeta_true_);
  return ok;
}

real_t LinearQuasiStaticSlipReferentialProblem::BlockNullPairResidual(
    const Vector& us_true, const Vector& uf_true) {
  EnsureOperator();
  real_t a_max = 0.0;
#ifdef MFEM_USE_MPI
  if (pfes_) {
    auto* hyp = A_.As<HypreParMatrix>();
    SparseMatrix diag, offd;
    HYPRE_BigInt* cmap = nullptr;
    hyp->GetDiag(diag);
    hyp->GetOffd(offd, cmap);
    real_t local = std::max(diag.MaxNorm(), offd.MaxNorm());
    MPI_Allreduce(&local, &a_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                  pfes_->GetComm());
  } else
#endif
  {
    a_max = A_.As<SparseMatrix>()->MaxNorm();
  }
  const real_t norm = std::sqrt(Dot(us_true, us_true) + Dot(uf_true, uf_true));

  // Physical residual (penalty and eps Q excluded).
  BlockVector r(offsets3_);
  op_A00_->Mult(us_true, r.GetBlock(0));
  op_A01_->AddMult(uf_true, r.GetBlock(0));
  op_A10_->Mult(us_true, r.GetBlock(1));
  op_A11_->AddMult(uf_true, r.GetBlock(1));
  op_A20_->Mult(us_true, r.GetBlock(2));
  op_A21_->AddMult(uf_true, r.GetBlock(2));
  return std::sqrt(Dot(r, r)) / (a_max * std::max(norm, real_t{1e-300}));
}

std::vector<real_t>
LinearQuasiStaticSlipReferentialProblem::SlipRigidPairResiduals() {
  EnsureOperator();
  std::vector<real_t> out;
  BlockVector nb(offsets3_);
  for (int i = 0; i < projector3_->Size(); i++) {
    static_cast<Vector&>(nb) = projector3_->Basis(i);
    // Skip the 2-D potential constant: it is not a (u_s, u_f) pair.
    if (nb.GetBlock(0).Norml2() == 0.0 && nb.GetBlock(1).Norml2() == 0.0) {
      continue;
    }
    out.push_back(BlockNullPairResidual(nb.GetBlock(0), nb.GetBlock(1)));
  }
  return out;
}

void LinearQuasiStaticSlipReferentialProblem::RegisterFields(
    DataCollection& dc) {
  LinearQuasiStaticReferentialProblem::RegisterFields(dc);
  dc.RegisterField("fluid_displacement", u_f_.get());
}

}  // namespace mfemElasticity
