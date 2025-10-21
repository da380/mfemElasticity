
#include "mfemElasticity/mesh.hpp"
  

namespace mfemElasticity {

mfem::Array<int> ExternalBoundaryMarker(mfem::Mesh* mesh) {
  auto bdr_marker = mfem::Array<int>(mesh->bdr_attributes.Max());
  bdr_marker = 0;
  mesh->MarkExternalBoundaries(bdr_marker);
  return bdr_marker;
}

mfem::Array<int> AllDomainsMarker(mfem::Mesh* mesh) {
  auto dom_marker = mfem::Array<int>(mesh->attributes.Max());
  dom_marker = 1;
  return dom_marker;
}

mfem::Array<int> AllBoundariesMarker(mfem::Mesh* mesh) {
  auto bdr_marker = mfem::Array<int>(mesh->bdr_attributes.Max());
  bdr_marker = 1;
  return bdr_marker;
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0) {
  using namespace mfem;

  const auto rtol = 1e-6;

  auto dim = mesh->Dimension();
  auto fec = H1_FECollection(1, dim);
  auto fes = FiniteElementSpace(mesh, &fec);

  auto radius = real_t{-1};
  auto x = Vector(dim);
  auto found = 0;
  auto same = 1;
  for (auto i = 0; i < mesh->GetNBE(); i++) {
    if (same == 0) break;
    const auto attr = mesh->GetBdrAttribute(i);

    if (bdr_marker[attr - 1] == 1) {
      found = 1;
      const auto* fe = fes.GetBE(i);
      auto* Trans = fes.GetBdrElementTransformation(i);
      const auto& ir = fe->GetNodes();
      for (auto j = 0; j < ir.GetNPoints(); j++) {
        const auto& ip = ir.IntPoint(j);
        Trans->SetIntPoint(&ip);
        Trans->Transform(ip, x);
        auto d = x.DistanceTo(x0);
        if (radius < 0) {
          radius = d;
        } else {
          same = static_cast<int>(std::abs(radius - d) < rtol * radius);
          if (same == 0) break;
        }
      }
    }
  }
  return {found, same, radius};
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, mfem::Array<int>&& bdr_marker, const mfem::Vector& x0) {
  return SphericalBoundaryRadius(mesh, bdr_marker, x0);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker) {
  auto x0 = mfem::Vector(mesh->Dimension());
  x0 = 0.0;
  return SphericalBoundaryRadius(mesh, bdr_marker, x0);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, mfem::Array<int>&& bdr_marker) {
  return SphericalBoundaryRadius(mesh, bdr_marker);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Vector& x0) {
  return SphericalBoundaryRadius(mesh, ExternalBoundaryMarker(mesh), x0);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(mfem::Mesh* mesh) {
  return SphericalBoundaryRadius(mesh, ExternalBoundaryMarker(mesh));
}

#ifdef MFEM_USE_MPI
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0) {
  using namespace mfem;

  const auto rtol = 1e-6;
  auto comm = mesh->GetComm();
  auto rank = mesh->GetMyRank();
  auto size = mesh->GetNRanks();

  auto [local_found, local_same, local_radius] =
      SphericalBoundaryRadius(dynamic_cast<Mesh*>(mesh), bdr_marker, x0);

  real_t radius;
  auto found = 0;
  auto same = 1;

  if (rank == 0) {
    auto founds = std::vector<int>(size);
    auto sames = std::vector<int>(size);
    auto radii = std::vector<real_t>(size);

    MPI_Gather(&local_found, 1, MPI_INT, founds.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&local_same, 1, MPI_INT, sames.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&local_radius, 1, MFEM_MPI_REAL_T, radii.data(), 1,
               MFEM_MPI_REAL_T, 0, comm);

    for (auto i = 0; i < size; i++) {
      if (founds[i] == 1 && sames[i] == 1) {
        found = 1;
        radius = radii[i];
        break;
      }
    }

    for (auto i = 0; i < size; i++) {
      if (founds[i] == 1 && sames[i] == 1) {
        if (std::abs(radius - radii[i]) > rtol * radius) {
          same = 0;
          break;
        }
      }
    }

  } else {
    MPI_Gather(&local_found, 1, MPI_INT, nullptr, 0, MPI_INT, 0, comm);
    MPI_Gather(&local_same, 1, MPI_INT, nullptr, 0, MPI_INT, 0, comm);
    MPI_Gather(&local_radius, 1, MFEM_MPI_REAL_T, nullptr, 0, MFEM_MPI_REAL_T,
               0, comm);
  }

  MPI_Bcast(&found, 1, MPI_INT, 0, comm);
  MPI_Bcast(&same, 1, MPI_INT, 0, comm);
  MPI_Bcast(&radius, 1, MFEM_MPI_REAL_T, 0, comm);

  return {found, same, radius};
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, mfem::Array<int>&& bdr_marker,
    const mfem::Vector& x0) {
  return SphericalBoundaryRadius(mesh, bdr_marker, x0);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker) {
  auto x0 = mfem::Vector(mesh->Dimension());
  x0 = 0.0;
  return SphericalBoundaryRadius(mesh, bdr_marker, x0);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, mfem::Array<int>&& bdr_marker) {
  auto x0 = mfem::Vector(mesh->Dimension());
  x0 = 0.0;
  return SphericalBoundaryRadius(mesh, bdr_marker, x0);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Vector& x0) {
  auto bdr_marker = ExternalBoundaryMarker(mesh);
  return SphericalBoundaryRadius(mesh, bdr_marker, x0);
}

std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh) {
  auto bdr_marker = ExternalBoundaryMarker(mesh);
  return SphericalBoundaryRadius(mesh, bdr_marker);
}

#endif

mfem::Vector MeshCentroid(mfem::Mesh* mesh, mfem::Array<int>& dom_marker,
                          int order) {
  using namespace mfem;
  auto dim = mesh->Dimension();
  auto x0 = Vector(dim);
  x0 = 0.0;

  auto fec = L2_FECollection(order, dim);
  auto fes = FiniteElementSpace(mesh, &fec);

  auto c = ConstantCoefficient(1);
  auto f = LinearForm(&fes);
  f.AddDomainIntegrator(new DomainLFIntegrator(c), dom_marker);
  f.Assemble();

  auto u = GridFunction(&fes);
  u.ProjectCoefficient(c);
  auto m = f(u);

  for (auto i = 0; i < dim; i++) {
    auto f = LinearForm(&fes);
    auto c = FunctionCoefficient([i](const Vector& x) { return x[i]; });
    f.AddDomainIntegrator(new DomainLFIntegrator(c), dom_marker);
    f.Assemble();
    x0(i) = f(u) / m;
  }

  return x0;
}

mfem::Vector MeshCentroid(mfem::Mesh* mesh, mfem::Array<int>&& dom_marker,
                          int order) {
  return MeshCentroid(mesh, dom_marker, order);
}

mfem::Vector MeshCentroid(mfem::Mesh* mesh, int order) {
  auto dom_marker = AllDomainsMarker(mesh);
  return MeshCentroid(mesh, dom_marker, order);
}

#ifdef MFEM_USE_MPI

mfem::Vector MeshCentroid(mfem::ParMesh* mesh, mfem::Array<int>& dom_marker,
                          int order) {
  using namespace mfem;
  auto dim = mesh->Dimension();
  auto x0 = Vector(dim);
  x0 = 0.0;

  auto fec = L2_FECollection(order, dim);
  auto fes = ParFiniteElementSpace(mesh, &fec);

  auto c = ConstantCoefficient(1);
  auto f = ParLinearForm(&fes);
  f.AddDomainIntegrator(new DomainLFIntegrator(c), dom_marker);
  f.Assemble();

  auto u = ParGridFunction(&fes);
  u.ProjectCoefficient(c);
  auto m = f(u);

  for (auto i = 0; i < dim; i++) {
    auto f = ParLinearForm(&fes);
    auto c = FunctionCoefficient([i](const Vector& x) { return x[i]; });
    f.AddDomainIntegrator(new DomainLFIntegrator(c), dom_marker);
    f.Assemble();
    x0(i) = f(u) / m;
  }

  return x0;
}

mfem::Vector MeshCentroid(mfem::ParMesh* mesh, mfem::Array<int>&& dom_marker,
                          int order) {
  return MeshCentroid(mesh, dom_marker, order);
}

mfem::Vector MeshCentroid(mfem::ParMesh* mesh, int order) {
  auto dom_marker = AllDomainsMarker(mesh);
  return MeshCentroid(mesh, dom_marker, order);
}
#endif

void SphericalMeshHelper::SetBoundaryMarker(mfem::Mesh* mesh) {
  _x0 = MeshCentroid(mesh);
  _bdr_marker = ExternalBoundaryMarker(mesh);
  auto [found, same, radius] = SphericalBoundaryRadius(mesh, _bdr_marker, _x0);
  assert(found == 1 && same == 1);
  _bdr_radius = radius;
}

#ifdef MFEM_USE_MPI
void SphericalMeshHelper::SetBoundaryMarker(mfem::ParMesh* mesh) {
  _x0 = MeshCentroid(mesh);
  _bdr_marker = ExternalBoundaryMarker(mesh);
  auto [found, same, radius] = SphericalBoundaryRadius(mesh, _bdr_marker, _x0);
  assert(found == 1 && same == 1);
  _bdr_radius = radius;
}

#endif

//new ZY contributions
LonLatField::LonLatField(std::vector<double> lons, std::vector<double> lats)
    : _lons(std::move(lons)), _lats(std::move(lats)),
      _nlon(static_cast<int>(_lons.size())), _nlat(static_cast<int>(_lats.size())) {}

size_t LonLatField::Idx(int i, int j) const {
    return static_cast<size_t>(j) * static_cast<size_t>(_nlon) + static_cast<size_t>(i);
}

double LonLatField::NorthPole(const std::vector<double>& field) const {
    const int j0 = _nlat - 2, j1 = _nlat - 1;
    const double y0 = _lats[j0], y1 = _lats[j1];
    const double dy = y1 - y0;
    const double t  = (90.0 - y1) / dy + 1.0;
    double sum = 0.0;
    for (int i = 0; i < _nlon; ++i) {
        const double v0 = field[Idx(i, j0)];
        const double v1 = field[Idx(i, j1)];
        sum += v0 * (1.0 - t) + v1 * t;
    }
    return sum / _nlon;
}

double LonLatField::SouthPole(const std::vector<double>& field) const {
    const int j0 = 0, j1 = 1;
    const double y0 = _lats[j0], y1 = _lats[j1];
    const double dy = y1 - y0;
    const double t  = (-90.0 - y0) / dy;
    double sum = 0.0;
    for (int i = 0; i < _nlon; ++i) {
        const double v0 = field[Idx(i, j0)];
        const double v1 = field[Idx(i, j1)];
        sum += v0 * (1.0 - t) + v1 * t;
    }
    return sum / _nlon;
}

double LonLatField::Bilerp(const std::vector<double>& field, double lon, double lat) const {
    if (_nlon <= 1 || _nlat <= 1)
        throw std::runtime_error("LonLatField::bilerp requires nlon>1 and nlat>1");

    if (lat >  90.0) lat =  90.0;
    if (lat < -90.0) lat = -90.0;

    {
        double x = std::fmod(lon + 180.0, 360.0);
        if (x < 0.0) x += 360.0;
        lon = x - 180.0;
    }

    const double lonMin = _lons.front();
    const double lonMax = _lons.back();

    int i0, i1;
    double a;

    if (lon >= lonMin && lon <= lonMax) {
        int i_hi = int(std::lower_bound(_lons.begin(), _lons.end(), lon) - _lons.begin());
        if (i_hi == 0) { i0 = 0; i1 = 1; }
        else if (i_hi >= _nlon) { i0 = _nlon - 2; i1 = _nlon - 1; }
        else { i0 = i_hi - 1; i1 = i_hi; }
        const double x0 = _lons[i0], x1 = _lons[i1];
        a = (x1 != x0) ? (lon - x0) / (x1 - x0) : 0.0;
    } else {
        i0 = _nlon - 1;
        i1 = 0;
        const double seamWidth = (lonMin + 360.0) - lonMax;
        if (lon > lonMax) a = (lon - lonMax) / seamWidth;
        else a = ((lon + 360.0) - lonMax) / seamWidth;
    }

    if (lat > _lats.back()) {
        const int jt = _nlat - 1;
        const double y0 = _lats[jt];
        const double vTop = (1.0 - a) * field[Idx(i0, jt)] + a * field[Idx(i1, jt)];
        const double den = (90.0 - y0);
        const double t = (lat - y0) / den;
        return (1.0 - t) * vTop + t * NorthPole(field);
    }

    if (lat < _lats.front()) {
        const int jb = 0;
        const double y1 = _lats[jb];
        const double vBottom = (1.0 - a) * field[Idx(i0, jb)] + a * field[Idx(i1, jb)];
        const double den = (y1 - (-90.0));
        const double t = (lat - (-90.0)) / den;
        return (1.0 - t) * SouthPole(field) + t * vBottom;
    }

    int j_hi = int(std::lower_bound(_lats.begin(), _lats.end(), lat) - _lats.begin());
    int j0, j1;
    if (j_hi == 0) { j0 = 0; j1 = 1; }
    else if (j_hi >= _nlat) { j0 = _nlat - 2; j1 = _nlat - 1; }
    else { j0 = j_hi - 1; j1 = j_hi; }

    const double y0 = _lats[j0], y1 = _lats[j1];
    const double b  = (y1 != y0) ? (lat - y0) / (y1 - y0) : 0.0;

    const double f00 = field[Idx(i0, j0)];
    const double f10 = field[Idx(i1, j0)];
    const double f01 = field[Idx(i0, j1)];
    const double f11 = field[Idx(i1, j1)];

    const double w00 = (1.0 - a) * (1.0 - b);
    const double w10 = a * (1.0 - b);
    const double w01 = (1.0 - a) * b;
    const double w11 = a * b;

    return f00 * w00 + f10 * w10 + f01 * w01 + f11 * w11;
}

PREMModel::PREMModel(const std::string& fileName,
                     double Rref,
                     double buffer_ratio,
                     int    ignored_layers)
    : _Rref(Rref),
      _buffer_ratio(buffer_ratio),
      _ignored_layers(ignored_layers)
{
    if (_buffer_ratio < 0.0)
        throw std::invalid_argument("buffer_depth must be >= 0");

    std::ifstream file(fileName);
    if (!file)
        throw std::runtime_error("Unable to open PREM file: " + fileName);

    std::string line;
    bool dataStarted = false;
    double prevR = std::numeric_limits<double>::quiet_NaN();

    auto try_parse_line = [&](const std::string& ln){
        std::istringstream iss(ln);
        double r, density, pWave, sWave, bulkM, shearM;
        if (!(iss >> r >> density >> pWave >> sWave >> bulkM >> shearM)) {
            throw std::runtime_error("PREM bad line: " + ln);
        }

        if (!std::isnan(prevR) && std::abs(r - prevR)/prevR < 1e-6) {
            radii.push_back(r);
        }
        prevR = r;
    };

    while (std::getline(file, line)) {
        if (!dataStarted) {
            std::istringstream probe(line);
            std::string firstTok;
            if (!(probe >> firstTok)) continue;
            if (firstTok == "0.") { dataStarted = true; try_parse_line(line); }
        } else {
            try_parse_line(line);
        }
    }

    if (_ignored_layers < 0 || static_cast<std::size_t>(_ignored_layers) > radii.size())
    {
        throw std::out_of_range("ignored_layers out of range");
    }

    const std::size_t keepN = radii.size() - static_cast<std::size_t>(_ignored_layers);
    radii_nd.reserve(keepN + 1);

    for (std::size_t i = 0; i < keepN; ++i)
        radii_nd.push_back(radii[i] / _Rref);

    radii_nd.push_back(1 + _buffer_ratio);
}

std::vector<double>& PREMModel::GetRadiiND() { return radii_nd; }
std::vector<double>& PREMModel::GetRadii() { return radii; }

Topography::Topography(const std::string& xyzFile, double Rref)
    : _Rref(Rref)
{
    std::vector<double> L, B, V;
    if (!LoadXYZ(xyzFile, L, B, V))
        throw std::runtime_error("Topography: cannot read " + xyzFile);
    for (double& v : V) v /= _Rref;
    BuildGrid(L, B, V);
}

Topography& Topography::operator+=(const Topography& other) {
    for (int j = 0; j < _grid.NLat(); ++j) {
        const double lat = LatAt(j);
        for (int i = 0; i < _grid.NLon(); ++i) {
            const double lon = LonAt(i);
            const double va  = _data[_grid.Idx(i, j)];
            const double vb  = other.Interp(lon, lat);
            if (std::isfinite(va) && std::isfinite(vb))
                _data[_grid.Idx(i, j)] = va + vb;
            else
                throw std::runtime_error("Infinite value in Topography::operator+=");
        }
    }
    return *this;
}

Topography operator+(const Topography& A, const Topography& B) {
    std::vector<double> V(static_cast<size_t>(A._grid.NLon()) * static_cast<size_t>(A._grid.NLat()),
                          std::numeric_limits<double>::quiet_NaN());
    for (int j = 0; j < A._grid.NLat(); ++j) {
        const double lat = A.LatAt(j);
        for (int i = 0; i < A._grid.NLon(); ++i) {
            const double lon = A.LonAt(i);
            const double va  = A._data[A._grid.Idx(i, j)];
            const double vb  = B.Interp(lon, lat);
            if (std::isfinite(va) && std::isfinite(vb))
                V[A._grid.Idx(i, j)] = va + vb;
            else
                throw std::runtime_error("Infinite value in Topography::operator+");
        }
    }
    return Topography(A._grid.Lons(), A._grid.Lats(), A._Rref, std::move(V));
}

double Topography::Interp(double lon, double lat) const {
    return _grid.Bilerp(_data, lon, lat);
}

int Topography::NLon() const { return _grid.NLon(); }
int Topography::NLat() const { return _grid.NLat(); }
const std::vector<double>& Topography::Lons() const { return _grid.Lons(); }
const std::vector<double>& Topography::Lats() const { return _grid.Lats(); }
const std::vector<double>& Topography::Data() const { return _data; }
double Topography::LonAt(int i) const { return _grid.LonAt(i); }
double Topography::LatAt(int j) const { return _grid.LatAt(j); }

double Topography::Mean() const {
    double sum = 0.0; size_t n = 0;
    for (double v : _data) if (std::isfinite(v)) { sum += v; ++n; }
    return n ? (sum / double(n)) : 0.0;
}

Topography::Topography(std::vector<double> lons, std::vector<double> lats, double Rref, std::vector<double> data)
    : _grid(LonLatField(std::move(lons), std::move(lats))),
      _Rref(Rref),
      _data(std::move(data)) {}

bool Topography::LoadXYZ(const std::string& file,
                         std::vector<double>& L, std::vector<double>& B, std::vector<double>& V)
{
    std::ifstream in(file);
    if (!in) return false;
    L.clear(); B.clear(); V.clear();
    double a, b, c;
    while (in >> a >> b >> c) { L.push_back(a); B.push_back(b); V.push_back(c); }
    return !L.empty();
}

void Topography::BuildGrid(const std::vector<double>& L,
                           const std::vector<double>& B,
                           const std::vector<double>& V)
{
    if (L.size() != B.size() || L.size() != V.size())
        throw std::runtime_error("Topography::buildGrid: xyz size mismatch");

    std::vector<double> lons = L, lats = B;
    std::sort(lons.begin(), lons.end()); lons.erase(std::unique(lons.begin(), lons.end()), lons.end());
    std::sort(lats.begin(), lats.end()); lats.erase(std::unique(lats.begin(), lats.end()), lats.end());

    _grid = LonLatField(std::move(lons), std::move(lats));
    _data.assign(static_cast<size_t>(_grid.NLon()) * static_cast<size_t>(_grid.NLat()),
                 std::numeric_limits<double>::quiet_NaN());

    const double tol = 1e-8;

    for (size_t k = 0; k < L.size(); ++k) {
        const double lon = L[k];
        const double lat = B[k];

        if (lon < _grid.Lons().front() - tol || lon > _grid.Lons().back() + tol ||
            lat < _grid.Lats().front() - tol || lat > _grid.Lats().back() + tol) {
            std::ostringstream oss;
            oss << "Topography::buildGrid: point (" << lon << ", " << lat << ") out of grid range";
            throw std::runtime_error(oss.str());
        }

        auto itx = std::lower_bound(_grid.Lons().begin(), _grid.Lons().end(), lon);
        auto ity = std::lower_bound(_grid.Lats().begin(), _grid.Lats().end(), lat);

        if (itx == _grid.Lons().end()) --itx;
        if (ity == _grid.Lats().end()) --ity;

        if (std::fabs(*itx - lon) > tol || std::fabs(*ity - lat) > tol) {
            std::ostringstream oss;
            oss << "Topography::buildGrid: (" << lon << "," << lat << ") not aligned to grid centers";
            throw std::runtime_error(oss.str());
        }

        const int i = static_cast<int>(itx - _grid.Lons().begin());
        const int j = static_cast<int>(ity - _grid.Lats().begin());
        _data[_grid.Idx(i, j)] = V[k];
    }
}

FieldRadialSurface::FieldRadialSurface(const LonLatField& grid, const std::vector<double>& r_field)
    : _grid(grid), _r_field(r_field) {}

double FieldRadialSurface::RadiusAt(double lon, double lat) const {
    return _grid.Bilerp(_r_field, lon, lat);
}

SpheroidalRadialSurface::SpheroidalRadialSurface(double r) : _r(r) {}
double SpheroidalRadialSurface::RadiusAt(double, double) const { return _r; }

EllipsoidalRadialSurface::EllipsoidalRadialSurface(double a, double b, double c) : _a(a), _b(b), _c(c) {}
double EllipsoidalRadialSurface::RadiusAt(double lon, double lat) const {
    const double L = Deg2Rad(lon), B = Deg2Rad(lat);
    const double nx = std::cos(B)*std::cos(L);
    const double ny = std::cos(B)*std::sin(L);
    const double nz = std::sin(B);
    const double denom = (nx*nx)/(_a*_a) + (ny*ny)/(_b*_b) + (nz*nz)/(_c*_c);
    return (denom > 0.0) ? 1.0 / std::sqrt(denom) : 0.0;
}

RadialMapping::RadialMapping(const std::vector<const Topography*>& topo, double topo_exag)
    : _topo(topo), _topo_exag(topo_exag) {}

double RadialMapping::InterpTopo(std::size_t i, double lon, double lat) const {
    return _topo[i]->Interp(lon, lat) * _topo_exag;
}

CubicBandLinearDecay::CubicBandLinearDecay(const std::vector<const Topography*>& topo,
                                           const std::vector<const RadialSurface*>& base,
                                           double decay,
                                           double topo_exag,
                                           std::size_t iInner,
                                           std::size_t iOuter)
    : RadialMapping(topo, topo_exag),
      _base(base), _decay(decay), _iInner(iInner), _iOuter(iOuter) {}

double CubicBandLinearDecay::Displacement(double r, double lon, double lat) const {
    const double rin   = _base[_iInner]->RadiusAt(lon, lat);
    const double rout  = _base[_iOuter]->RadiusAt(lon, lat);
    const double dInner = InterpTopo(_iInner, lon, lat);
    const double dOuter = InterpTopo(_iOuter, lon, lat);

    const double r_in_lo  = rin - _decay;
    const double r_mid_lo = rin;
    const double r_mid_hi = rout;
    const double r_out_hi = rout + _decay;

    if (r <= 0.0 || r < r_in_lo || r > r_out_hi) return 0.0;

    if (r <= r_mid_lo) {
        double t = (rin - r) / _decay;
        t = std::clamp(t, 0.0, 1.0);
        return dInner * (1.0 - t);
    } else if (r < r_mid_hi) {
        double t = (r - rin) / (rout - rin);
        t = std::clamp(t, 0.0, 1.0);
        const double w = 1.0 - t*t*(3.0 - 2.0*t);
        return w * dInner + (1.0 - w) * dOuter;
    } else {
        double t = (r - rout) / _decay;
        t = std::clamp(t, 0.0, 1.0);
        return dOuter * (1.0 - t);
    }
}




}  // namespace mfemElasticity
