

#pragma once

#include <optional>
#include <tuple>

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @brief Generates a marker array for the external boundary of a mesh.
 *
 * This function creates an `mfem::Array<int>` suitable for marking
 * all boundary elements that are part of the mesh's external boundary.
 * Boundary attributes are marked with \f$1\f$, non-boundary attributes with
 * \f$0\f$.
 *
 * @param mesh Pointer to the `mfem::Mesh` object.
 * @return An `mfem::Array<int>` where each entry corresponds to a
 * boundary attribute. The size of the array is `mesh->bdr_attributes.Size()`.
 */
mfem::Array<int> ExternalBoundaryMarker(mfem::Mesh* mesh);

/**
 * @brief Generates a marker array for all domain attributes of a mesh.
 *
 * This function creates an `mfem::Array<int>` that marks all existing
 * domain attributes in the mesh. Useful for selecting all elements in the
 * domain.
 *
 * @param mesh Pointer to the `mfem::Mesh` object.
 * @return An `mfem::Array<int>` where each entry corresponds to a
 * domain attribute. The size of the array is `mesh->attributes.Size()`.
 */
mfem::Array<int> AllDomainsMarker(mfem::Mesh* mesh);

/**
 * @brief Generates a marker array for all boundary attributes of a mesh.
 *
 * This function creates an `mfem::Array<int>` that marks all existing
 * boundary attributes in the mesh. Useful for selecting all boundary elements.
 *
 * @param mesh Pointer to the `mfem::Mesh` object.
 * @return An `mfem::Array<int>` where each entry corresponds to a
 * boundary attribute. The size of the array is `mesh->bdr_attributes.Size()`.
 */
mfem::Array<int> AllBoundariesMarker(mfem::Mesh* mesh);

/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius.
 *
 *
 * @param mesh Pointer to the `mfem::Mesh` object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @param x0 The origin (center) from which the radius is measured.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The radius found. Meaningful only if the first two
 *    return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0);

/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius (move version).
 *
 * This overload takes the `bdr_marker` by rvalue reference, allowing for
 * efficient passing of temporary marker arrays.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider (moved).
 * @param x0 The origin (center) from which the radius is measured.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, mfem::Array<int>&& bdr_marker, const mfem::Vector& x0);

/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius, using the mesh centroid as the origin.
 *
 * This overload automatically calculates the mesh centroid and uses it as the
 * origin `x0`.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker);

/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius, using the mesh centroid as the origin (move version).
 *
 * This overload takes `bdr_marker` by rvalue reference and uses the mesh
 * centroid as origin.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider (moved).
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, mfem::Array<int>&& bdr_marker);

/**
 * @brief Determines if the external boundary is spherical and returns its
 * radius from a given origin.
 *
 * This overload automatically marks the external boundary of the mesh and
 * uses it for the radius computation.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @param x0 The origin (center) from which the radius is measured.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::Mesh* mesh, const mfem::Vector& x0);

/**
 * @brief Determines if the external boundary is spherical and returns its
 * radius, using the mesh centroid as the origin.
 *
 * This overload automatically marks the external boundary and calculates the
 * mesh centroid to be used as the origin.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(mfem::Mesh* mesh);

#ifdef MFEM_USE_MPI

/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius for a parallel mesh.
 *
 * This function is the parallel counterpart of the serial
 * `SphericalBoundaryRadius` function. It considers the maximum radius across
 * all processors.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @param x0 The origin (center) from which the radius is measured.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The global radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0);

/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius for a parallel mesh (move version).
 *
 * This parallel overload takes the `bdr_marker` by rvalue reference.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider (moved).
 * @param x0 The origin (center) from which the radius is measured.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The global radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, mfem::Array<int>&& bdr_marker, const mfem::Vector& x0);

/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius for a parallel mesh, using the global mesh centroid as the origin.
 *
 * This parallel overload automatically calculates the global mesh centroid and
 * uses it as the origin.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The global radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker);
/**
 * @brief Determines if an indicated boundary is spherical and returns its
 * radius for a parallel mesh, using the global mesh centroid as the origin
 * (move version).
 *
 * This parallel overload takes `bdr_marker` by rvalue reference and uses the
 * global mesh centroid.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param bdr_marker An `mfem::Array<int>` marking which boundary attributes
 * (1 for inclusion, 0 for exclusion) to consider (moved).
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The global radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, mfem::Array<int>&& bdr_marker);
/**
 * @brief Determines if the external boundary is spherical and returns its
 * radius from a given origin for a parallel mesh.
 *
 * This parallel overload automatically marks the external boundary globally and
 * uses the provided origin.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param x0 The origin (center) from which the radius is measured.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The global radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Vector& x0);

/**
 * @brief Determines if the external boundary is spherical and returns its
 * radius from the global mesh centroid for a parallel mesh.
 *
 * This parallel overload automatically marks the external boundary globally and
 * calculates the global mesh centroid.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @return A `std::tuple` containing:
 * - `int`: Equal to 1 if the boundary is non-empty, 0 otherwise.
 * - `int`: Equal to 1 if the radii of all boundary points are (approximately)
 * equal.
 * - `mfem::real_t`: The global radius found. Meaningful only if the first two
 * return values equal 1.
 */
std::tuple<int, int, mfem::real_t> SphericalBoundaryRadius(mfem::ParMesh* mesh);
#endif

/**
 * @brief Computes the centroid of a mesh, optionally for a subset of domain
 * attributes.
 *
 * The centroid is computed by integrating the position vector over the
 * specified domain(s) and dividing by the total volume. The integration is
 * performed using a specified polynomial order for the quadrature rule.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @param dom_marker An `mfem::Array<int>` marking which domain attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @param order The polynomial order used for the quadrature rule during
 * integration.
 * @return An `mfem::Vector` representing the coordinates of the computed
 * centroid.
 */
mfem::Vector MeshCentroid(mfem::Mesh* mesh, mfem::Array<int>& dom_marker,
                          int order = 1);

/**
 * @brief Computes the centroid of a mesh (move version), optionally for a
 * subset of domain attributes.
 *
 * This overload takes the `dom_marker` by rvalue reference.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @param dom_marker An `mfem::Array<int>` marking which domain attributes
 * (1 for inclusion, 0 for exclusion) to consider (moved).
 * @param order The polynomial order used for the quadrature rule during
 * integration.
 * @return An `mfem::Vector` representing the coordinates of the computed
 * centroid.
 */
mfem::Vector MeshCentroid(mfem::Mesh* mesh, mfem::Array<int>&& dom_marker,
                          int order = 1);

/**
 * @brief Computes the centroid of the entire mesh.
 *
 * This overload computes the centroid considering all domain attributes.
 *
 * @param mesh Pointer to the mfem::Mesh object.
 * @param order The polynomial order used for the quadrature rule during
 * integration.
 * @return An `mfem::Vector` representing the coordinates of the computed
 * centroid.
 */
mfem::Vector MeshCentroid(mfem::Mesh* mesh, int order = 1);

#ifdef MFEM_USE_MPI
/**
 * @brief Computes the global centroid of a parallel mesh, optionally for a
 * subset of domain attributes.
 *
 * This parallel overload computes the centroid by accumulating contributions
 * from all processors.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param dom_marker An `mfem::Array<int>` marking which domain attributes
 * (1 for inclusion, 0 for exclusion) to consider.
 * @param order The polynomial order used for the quadrature rule during
 * integration.
 * @return An `mfem::Vector` representing the global coordinates of the computed
 * centroid.
 */
mfem::Vector MeshCentroid(mfem::ParMesh* mesh, mfem::Array<int>& dom_marker,
                          int order = 1);

/**
 * @brief Computes the global centroid of a parallel mesh (move version),
 * optionally for a subset of domain attributes.
 *
 * This parallel overload takes the `dom_marker` by rvalue reference.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param dom_marker An `mfem::Array<int>` marking which domain attributes
 * (1 for inclusion, 0 for exclusion) to consider (moved).
 * @param order The polynomial order used for the quadrature rule during
 * integration.
 * @return An `mfem::Vector` representing the global coordinates of the computed
 * centroid.
 */
mfem::Vector MeshCentroid(mfem::ParMesh* mesh, mfem::Array<int>&& dom_marker,
                          int order = 1);

/**
 * @brief Computes the global centroid of the entire parallel mesh.
 *
 * This parallel overload computes the centroid considering all domain
 * attributes across all processors.
 *
 * @param mesh Pointer to the mfem::ParMesh object.
 * @param order The polynomial order used for the quadrature rule during
 * integration.
 * @return An `mfem::Vector` representing the global coordinates of the computed
 * centroid.
 */
mfem::Vector MeshCentroid(mfem::ParMesh* mesh, int order = 1);
#endif

/**
 * @brief Struct providing utilities for a mesh with a spherical external
 * boundary.
 *
 * This helper struct encapsulates properties and methods relevant to meshes
 * that are known to have an external boundary that lies on a spherical surface.
 */
struct SphericalMeshHelper {
  /** @brief The radius of the spherical external boundary. */
  mfem::real_t _bdr_radius;
  /** @brief The center coordinates of the spherical boundary. */
  mfem::Vector _x0;
  /** @brief Marker array identifying the external boundary attributes. */
  mfem::Array<int> _bdr_marker;

  /**
   * @brief Determines and sets the external boundary marker for a serial mesh.
   *
   * This method populates `_bdr_marker`, `_bdr_radius`, and `_x0` by
   * analyzing the provided serial mesh.
   * @param mesh Pointer to the mfem::Mesh object.
   */
  void SetBoundaryMarker(mfem::Mesh* mesh);

#ifdef MFEM_USE_MPI
  /**
   * @brief Determines and sets the external boundary marker for a parallel
   * mesh.
   *
   * This method populates `_bdr_marker`, `_bdr_radius`, and `_x0` by
   * analyzing the provided parallel mesh, performing necessary MPI
   * communication to ensure global consistency.
   * @param mesh Pointer to the mfem::ParMesh object.
   */
  void SetBoundaryMarker(mfem::ParMesh* mesh);
#endif
};

//new contributions from ZY
#include <algorithm>
#include <cctype>     
#include <chrono>     
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>    
#include <sstream>
#include <string>
#include <vector>

//helpers
struct Timer {
    using clk = std::chrono::steady_clock;
    clk::time_point t0 = clk::now(), last = t0;
    void mark(const std::string& msg){
        auto now = clk::now();
        double dt = std::chrono::duration<double>(now-last).count();
        double tot= std::chrono::duration<double>(now-t0).count();
        last = now;
        std::cout<<std::fixed<<std::setprecision(3)
            <<"[TIMER] +"<<dt<<" s ("<<tot<<" s total): "<<msg<<"\n";
    }
};

inline std::vector<double> parseDoubles(const std::string &s){
    std::vector<double> v; std::istringstream iss(s); std::string tok;
    while (std::getline(iss, tok, '-')){
        tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
        if(!tok.empty()) v.push_back(std::stod(tok));
    }
    return v;
}

inline double deg2rad(double d){ return d * std::numbers::pi / 180.0; }
inline double rad2deg(double r){ return r * 180.0 / std::numbers::pi; }

//field on a lon-lat grid
class LonLatField {
public:
    LonLatField() = default;
    LonLatField(std::vector<double> lons, std::vector<double> lats);

    int nlon() const { return _nlon; }
    int nlat() const { return _nlat; }

    const std::vector<double>& lons() const { return _lons; }
    const std::vector<double>& lats() const { return _lats; }

    double lonAt(int i) const { return _lons[i]; }
    double latAt(int j) const { return _lats[j]; }

    size_t idx(int i, int j) const;

    double northPole(const std::vector<double>& field) const;
    double southPole(const std::vector<double>& field) const;
    double bilerp(const std::vector<double>& field, double lon, double lat) const;

private:
    std::vector<double> _lons, _lats;
    int _nlon = 0, _nlat = 0;
};

//Class for managing PREM data
//Lines are skipped until the linear starting with 0.
class PREMModel {
public:
    PREMModel(const std::string& fileName,
              double Rref,
              double buffer_ratio,
              int    ignored_layers = 0);

    std::vector<double>& getRadiiND();
    std::vector<double>& getRadii();

    ~PREMModel() = default;

private:
    double      _Rref;
    double      _buffer_ratio;
    int         _ignored_layers;

    std::vector<double> radii;     
    std::vector<double> radii_nd;
    std::vector<double> density_list, pWave_list, sWave_list, bulkM_list, shearM_list;
};

//each surface stored as a Topography class
class Topography {
public:
    Topography() = default;
    Topography(const Topography&) = default;
    Topography(Topography&&) noexcept = default;
    Topography& operator=(const Topography&) = default;
    Topography& operator=(Topography&&) noexcept = default;

    Topography(const std::string& xyzFile, double Rref = 1.0);

    Topography& operator+=(const Topography& other);
    friend Topography operator+(const Topography& A, const Topography& B);

    double interp(double lon, double lat) const;

    int nlon() const;
    int nlat() const;
    const std::vector<double>& lons() const;
    const std::vector<double>& lats() const;
    const std::vector<double>& data() const;

    double lonAt(int i) const;
    double latAt(int j) const;

    double mean() const;

private:
    LonLatField _grid;
    double _Rref = 1.0;
    std::vector<double> _data;

    Topography(std::vector<double> lons, std::vector<double> lats, double Rref, std::vector<double> data);

    static bool loadXYZ(const std::string& file,
                        std::vector<double>& L, std::vector<double>& B, std::vector<double>& V);

    void buildGrid(const std::vector<double>& L,
                   const std::vector<double>& B,
                   const std::vector<double>& V);
};

double meanRadiusOfSurface(int surfTag);

//RadialSurface hierarchy
class RadialSurface {
public:
    virtual ~RadialSurface() = default;
    virtual double radiusAt(double lon, double lat) const = 0;
};

class FieldRadialSurface final : public RadialSurface {
public:
    FieldRadialSurface(const LonLatField& grid, const std::vector<double>& r_field);
    double radiusAt(double lon, double lat) const override;
private:
    const LonLatField& _grid;
    const std::vector<double>& _r_field;
};

class SpheroidalRadialSurface final : public RadialSurface {
public:
    explicit SpheroidalRadialSurface(double r);
    double radiusAt(double, double) const override;
private:
    double _r;
};

class EllipsoidalRadialSurface final : public RadialSurface {
public:
    EllipsoidalRadialSurface(double a, double b, double c);
    double radiusAt(double lon, double lat) const override;
private:
    double _a, _b, _c;
};

//RadialMapping 
class RadialMapping {
public:
    RadialMapping(const std::vector<const Topography*>& topo, double topo_exag = 1.0);
    virtual ~RadialMapping() = default;
    virtual double displacement(double r, double lon, double lat) const = 0;

protected:
    double interpTopo(std::size_t i, double lon, double lat) const;

    const std::vector<const Topography*>& _topo;
    double _topo_exag;
};

class cubicBandLinearDecay final : public RadialMapping {
public:
    cubicBandLinearDecay(const std::vector<const Topography*>& topo,
                         const std::vector<const RadialSurface*>& base,
                         double decay,
                         double topo_exag = 1.0,
                         std::size_t iInner = 0,
                         std::size_t iOuter = 1);

    double displacement(double r, double lon, double lat) const override;

private:
    const std::vector<const RadialSurface*>& _base;
    double _decay = 0.0;
    std::size_t _iInner = 0;
    std::size_t _iOuter = 1;
};

//inline functions
void perturbAllNodes(const RadialMapping& mapping);

void tagLayersByRadius(const std::vector<int>& volTags,
                       const std::string& volPrefix = "volume_",
                       const std::string& surfPrefix = "surface_");

}  // namespace mfemElasticity
