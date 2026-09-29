/**
 * @file mappings.cpp
 * @brief Implementation of the mapping layer in mappings.hpp.
 */

#include "mfemElasticity/mappings.hpp"

namespace mfemElasticity {

using namespace mfem;

real_t Diffeomorphism::Jacobian(ElementTransformation& T,
                                const IntegrationPoint& ip) {
  EvalGradient(F_, T, ip);
  return F_.Det();
}

CallableDiffeomorphism::CallableDiffeomorphism(int dim, MapFunc xi,
                                               GradientFunc F)
    : Diffeomorphism(dim), xi_(std::move(xi)), grad_(std::move(F)), x_(dim) {}

void CallableDiffeomorphism::Eval(Vector& V, ElementTransformation& T,
                                  const IntegrationPoint& ip) {
  T.Transform(ip, x_);
  V.SetSize(vdim);
  xi_(x_, V);
}

void CallableDiffeomorphism::EvalGradient(DenseMatrix& F,
                                          ElementTransformation& T,
                                          const IntegrationPoint& ip) {
  T.Transform(ip, x_);
  F.SetSize(vdim);
  grad_(x_, F);
}

RadialDiffeomorphism::RadialDiffeomorphism(int dim, Coefficient& f,
                                           VectorCoefficient& grad_f)
    : Diffeomorphism(dim), f_(&f), grad_f_(&grad_f), x_(dim), g_(dim) {
  MFEM_VERIFY(grad_f.GetVDim() == dim, "RadialDiffeomorphism: grad_f vdim.");
}

RadialDiffeomorphism::RadialDiffeomorphism(
    int dim, std::function<real_t(real_t)> f, std::function<real_t(real_t)> df)
    : Diffeomorphism(dim), fr_(std::move(f)), dfr_(std::move(df)), x_(dim),
      g_(dim) {}

real_t RadialDiffeomorphism::Scalar(ElementTransformation& T,
                                    const IntegrationPoint& ip) {
  return f_ ? f_->Eval(T, ip) : fr_(x_.Norml2());
}

void RadialDiffeomorphism::Eval(Vector& V, ElementTransformation& T,
                                const IntegrationPoint& ip) {
  T.Transform(ip, x_);
  V.SetSize(vdim);
  V = x_;
  V *= Scalar(T, ip);
}

void RadialDiffeomorphism::EvalGradient(DenseMatrix& F,
                                        ElementTransformation& T,
                                        const IntegrationPoint& ip) {
  T.Transform(ip, x_);
  const real_t f = Scalar(T, ip);
  if (grad_f_) {
    grad_f_->Eval(g_, T, ip);
  } else {
    const real_t r = x_.Norml2();
    g_ = x_;
    g_ *= r > 0.0 ? dfr_(r) / r : 0.0;
  }
  F.SetSize(vdim);
  for (int i = 0; i < vdim; i++) {
    for (int A = 0; A < vdim; A++) {
      F(i, A) = x_(i) * g_(A);
    }
    F(i, i) += f;
  }
}

GridFunctionDiffeomorphism::GridFunctionDiffeomorphism(const GridFunction& h)
    : Diffeomorphism(h.VectorDim()), h_(&h), x_(h.VectorDim()) {
  MFEM_VERIFY(
      h.VectorDim() == h.FESpace()->GetMesh()->SpaceDimension(),
      "GridFunctionDiffeomorphism: the displacement must have vdim equal "
      "to the space dimension.");
}

GridFunctionDiffeomorphism::GridFunctionDiffeomorphism(
    std::unique_ptr<FiniteElementCollection> fec,
    std::unique_ptr<FiniteElementSpace> fes, std::unique_ptr<GridFunction> h)
    : Diffeomorphism(h->VectorDim()),
      owned_fec_(std::move(fec)),
      owned_fes_(std::move(fes)),
      owned_h_(std::move(h)),
      h_(owned_h_.get()),
      x_(vdim) {}

void GridFunctionDiffeomorphism::Eval(Vector& V, ElementTransformation& T,
                                      const IntegrationPoint& ip) {
  T.Transform(ip, x_);
  h_->GetVectorValue(T, ip, V);
  V += x_;
}

void GridFunctionDiffeomorphism::EvalGradient(DenseMatrix& F,
                                              ElementTransformation& T,
                                              const IntegrationPoint& ip) {
  T.SetIntPoint(&ip);
  F.SetSize(vdim);
  h_->GetVectorGradient(T, F);
  for (int i = 0; i < vdim; i++) {
    F(i, i) += 1.0;
  }
}

real_t TransformedFunctionCoefficient::Eval(ElementTransformation& T,
                                            const IntegrationPoint& ip) {
  real_t data[3];
  Vector y(data, 3);
  xi_->Eval(y, T, ip);
  return f_(y);
}

void TransformedVectorFunctionCoefficient::Eval(Vector& V,
                                                ElementTransformation& T,
                                                const IntegrationPoint& ip) {
  xi_->Eval(y_, T, ip);
  V.SetSize(vdim);
  f_(y_, V);
}

real_t JacobianCoefficient::Eval(ElementTransformation& T,
                                 const IntegrationPoint& ip) {
  return xi_->Jacobian(T, ip);
}

void PullbackDiffusionCoefficient::Eval(DenseMatrix& K,
                                        ElementTransformation& T,
                                        const IntegrationPoint& ip) {
  xi_->EvalGradient(F_, T, ip);
  const real_t J = F_.Det();
  F_.Invert();
  K.SetSize(F_.Height());
  MultABt(F_, F_, K);
  K *= J;
}

void PullbackGradientCoefficient::Eval(Vector& V, ElementTransformation& T,
                                       const IntegrationPoint& ip) {
  v_->Eval(w_, T, ip);
  xi_->EvalGradient(F_, T, ip);
  F_.Invert();
  V.SetSize(vdim);
  F_.MultTranspose(w_, V);
}

namespace {

// The interpolation space of a mesh's geometry: its nodal collection,
// order and ordering, or H1 order 1 for a mesh without nodes. Shared by
// the serial and parallel Interpolate.
std::unique_ptr<FiniteElementCollection> GeometricCollection(
    const Mesh& mesh, int& ordering) {
  const GridFunction* nodes = mesh.GetNodes();
  if (nodes) {
    ordering = nodes->FESpace()->GetOrdering();
    return std::unique_ptr<FiniteElementCollection>(
        FiniteElementCollection::New(nodes->FESpace()->FEColl()->Name()));
  }
  ordering = Ordering::byVDIM;
  return std::make_unique<H1_FECollection>(1, mesh.Dimension());
}

// h = interpolant of xi minus interpolant of the identity (the latter
// exact, the nodal space containing linears).
void SetDisplacement(GridFunction& h, Diffeomorphism& xi) {
  h.ProjectCoefficient(xi);
  GridFunction id(h.FESpace());
  VectorFunctionCoefficient identity(
      h.VectorDim(), [](const Vector& x, Vector& y) { y = x; });
  id.ProjectCoefficient(identity);
  h -= id;
}

}  // namespace

GridFunctionDiffeomorphism Interpolate(Diffeomorphism& xi, Mesh& mesh) {
  int ordering;
  auto fec = GeometricCollection(mesh, ordering);
  auto fes = std::make_unique<FiniteElementSpace>(
      &mesh, fec.get(), mesh.SpaceDimension(), ordering);
  auto h = std::make_unique<GridFunction>(fes.get());
  SetDisplacement(*h, xi);
  return GridFunctionDiffeomorphism(std::move(fec), std::move(fes),
                                    std::move(h));
}

Mesh MappedMesh(const Mesh& mesh, Diffeomorphism& xi) {
  auto mapped = Mesh(mesh);
  if (mapped.GetNodes() == nullptr) {
    mapped.SetCurvature(1);
  }
  mapped.Transform(xi);
  return mapped;
}

real_t MaxIdentityDeviation(Diffeomorphism& xi, Mesh& mesh,
                            const Array<int>& bdr_marker) {
  const int sdim = mesh.SpaceDimension();
  Vector x(sdim), y(sdim);
  real_t dev = 0.0;
  for (int i = 0; i < mesh.GetNBE(); i++) {
    const int attr = mesh.GetBdrAttribute(i);
    if (bdr_marker[attr - 1] == 0) {
      continue;
    }
    ElementTransformation* T = mesh.GetBdrElementTransformation(i);
    const IntegrationRule& ir =
        IntRules.Get(T->GetGeometryType(), 2 * T->OrderJ() + 2);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T->Transform(ip, x);
      xi.Eval(y, *T, ip);
      y -= x;
      dev = std::max(dev, y.Norml2());
    }
  }
  return dev;
}

#ifdef MFEM_USE_MPI

GridFunctionDiffeomorphism Interpolate(Diffeomorphism& xi, ParMesh& mesh) {
  int ordering;
  auto fec = GeometricCollection(mesh, ordering);
  auto fes = std::make_unique<ParFiniteElementSpace>(
      &mesh, fec.get(), mesh.SpaceDimension(), ordering);
  auto h = std::make_unique<ParGridFunction>(fes.get());
  SetDisplacement(*h, xi);
  return GridFunctionDiffeomorphism(std::move(fec), std::move(fes),
                                    std::move(h));
}

ParMesh MappedMesh(const ParMesh& mesh, Diffeomorphism& xi) {
  auto mapped = ParMesh(mesh);
  if (mapped.GetNodes() == nullptr) {
    mapped.SetCurvature(1);
  }
  mapped.Transform(xi);
  return mapped;
}

real_t MaxIdentityDeviation(Diffeomorphism& xi, ParMesh& mesh,
                            const Array<int>& bdr_marker) {
  const real_t local =
      MaxIdentityDeviation(xi, static_cast<Mesh&>(mesh), bdr_marker);
  real_t global;
  MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                mesh.GetComm());
  return global;
}

#endif

}  // namespace mfemElasticity
