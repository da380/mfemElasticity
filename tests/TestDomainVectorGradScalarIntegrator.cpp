TEST_P(InterpolatorTest, DomainVectorGradScalarIntegratorScalarCoefficient) {
  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &H1);
  auto vector_fes = FiniteElementSpace(&mesh, &L2, dim);

  auto b = mfem::MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(new mfemElasticity::DomainVectorGradScalarIntegrator());
  b.Assemble();

  auto u = mfem::GridFunction(&scalar_fes);
  auto uF = mfem::FunctionCoefficient(
      [](const mfem::Vector& x) { return x(0) * x(1) * x(1); });
  u.ProjectCoefficient(uF);

  auto v = mfem::GridFunction(&vector_fes);
  auto vF = mfem::VectorFunctionCoefficient(
      dim, [](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(x.Size());
        y = 1.;
      });
  v.ProjectCoefficient(vF);

  auto w = mfem::GridFunction(&vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto l = mfem::LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(vF));
  l.Assemble();
  auto value2 = l * u;

  auto error = std::abs(value2 - value1) / std::abs(value1);
  EXPECT_TRUE(error < 1.e-6);
}

TEST_P(InterpolatorTest, DomainVectorGradScalarIntegratorVectorCoefficient) {
  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &H1);
  auto vector_fes = FiniteElementSpace(&mesh, &L2, dim);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> distrib(0, 1);

  auto a = Vector(dim);
  for (auto j = 0; j < dim; j++) {
    a(j) = distrib(gen);
  }
  auto qv = VectorConstantCoefficient(a);

  auto b = mfem::MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(
      new mfemElasticity::DomainVectorGradScalarIntegrator(qv));
  b.Assemble();

  auto u = mfem::GridFunction(&scalar_fes);
  auto uF = mfem::FunctionCoefficient(
      [](const mfem::Vector& x) { return x.Norml2(); });
  u.ProjectCoefficient(uF);

  auto v = mfem::GridFunction(&vector_fes);
  auto vF = mfem::VectorFunctionCoefficient(
      dim, [](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(x.Size());
        y = x;
      });
  v.ProjectCoefficient(vF);

  auto w = mfem::GridFunction(&vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto l = mfem::LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(vF));
  l.Assemble();
  auto value2 = l * u;

  auto error = std::abs(value2 - value1) / std::abs(value1);
  EXPECT_TRUE(error < 1.e-6);
}

TEST_P(InterpolatorTest, DomainVectorGradScalarIntegratorMatrixCoefficient) {
  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &H1);
  auto vector_fes = FiniteElementSpace(&mesh, &L2, dim);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> distrib(0, 1);

  auto A = DenseMatrix(dim);
  for (auto j = 0; j < dim; j++) {
    for (auto i = 0; i < dim; i++) {
      A(i, j) = distrib(gen);
    }
  }
  auto qm = MatrixConstantCoefficient(A);

  auto b = mfem::MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(
      new mfemElasticity::DomainVectorGradScalarIntegrator(qm));
  b.Assemble();

  auto u = mfem::GridFunction(&scalar_fes);
  auto uF = mfem::FunctionCoefficient(
      [](const mfem::Vector& x) { return x(0) * x(1) * x(1); });
  u.ProjectCoefficient(uF);

  auto v = mfem::GridFunction(&vector_fes);
  auto vF = mfem::VectorFunctionCoefficient(
      dim, [](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(x.Size());
        y = 1.;
      });
  v.ProjectCoefficient(vF);

  auto w = mfem::GridFunction(&vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto l = mfem::LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(vF));
  l.Assemble();
  auto value2 = l * u;

  auto error = std::abs(value2 - value1) / std::abs(value1);
  EXPECT_TRUE(error < 1.e-6);
}
