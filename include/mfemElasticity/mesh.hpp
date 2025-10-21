

#pragma once

#include <algorithm>   
#include <cmath>   
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>  
#include <sstream>
#include <limits>      
#include <stdexcept>  
#include <cctype>
#include <tuple>     
#include <numbers>
#include <chrono>
#include <utility>
#include <optional>


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

/**
 * @brief A global timer for convenience.
 */
struct Timer {
    using clk = std::chrono::steady_clock;
    /** @brief Initial time when the timer is constructed. */
    clk::time_point t0 = clk::now(); 
    /** @brief Last time stop. */
    clk::time_point last = t0;
    /**
     * @brief Output the time since the previous mark and total time since construction.
     *
     * Prints in the format:
     * `[TIMER] +<dt> s (<tot> s total): <msg>`
     *
     * @param msg A message describing the section that just completed.
     *
     * @note Uses a monotonic clock (steady_clock). Units are seconds.
     * @post Updates the internal @ref last timestamp to 'now'.
     */
    void Mark(const std::string& msg){
        auto now = clk::now();
        mfem::real_t dt = std::chrono::duration<mfem::real_t>(now-last).count();
        mfem::real_t tot= std::chrono::duration<mfem::real_t>(now-t0).count();
        last = now;
        std::cout<<std::fixed<<std::setprecision(3)
            <<"[TIMER] +"<<dt<<" s ("<<tot<<" s total): "<<msg<<"\n";
    }
};

/**
 * @brief Parse doubles separated by `-` (e.g. mesh sizes).
 *
 * Splits on '-' characters, trims whitespace in each token, and converts non-empty tokens to double.
 *
 * @param s A string in CLI (e.g., "1.0-0.5-0.25").
 * @return A std::vector<mfem::real_t> of parsed double values.
 */
inline std::vector<mfem::real_t> ParseDoubles(const std::string &s){
    std::vector<mfem::real_t> v; std::istringstream iss(s); std::string tok;
    while (std::getline(iss, tok, '-')){
        tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
        if(!tok.empty()) v.push_back(static_cast<mfem::real_t>(std::stod(tok)));
    }
    return v;
}

/**
 * @brief An inline function transforming degree to radian.
 *
 * @param d Degree value.
 * @return r Radian value.
 *
 * @note Uses std::numbers::pi. Input/Output are mfem::real_t.
 */
inline mfem::real_t Deg2Rad(mfem::real_t d){ return d * std::numbers::pi / 180.0; }

/**
 * @brief An inline function transforming radian to degree.
 *
 * @param r Radian value.
 * @return Degree value.
 *
 * @note Uses std::numbers::pi. Input/Output are mfem::real_t.
 */
inline mfem::real_t Rad2Deg(mfem::real_t r){ return r * 180.0 / std::numbers::pi; }

/**
 * @brief A class representing the lon-lat grid.
 *
 * Stores 1D arrays of longitudes and latitudes and provides index mapping
 * and bilinear interpolation support for fields sampled on the tensor grid.
 */ 
class LonLatField {
public:
    /** @brief Default-constructed empty grid. */
    LonLatField() = default;

    /** 
     * @brief Constructs a LonLatFied with lists of lon- and lat- coordinates.
     *
     * @param lons Longitudes in degrees.
     * @param lats Latitudes in degrees.
     */
    LonLatField(std::vector<mfem::real_t> lons, std::vector<mfem::real_t> lats);

    /** @brief Number of longitude nodes. */
    int NLon() const { return _nlon; }
    /** @brief Number of latitude nodes. */
    int NLat() const { return _nlat; }

    /** @brief Read-only access to longitudes. */
    const std::vector<mfem::real_t>& Lons() const { return _lons; }
    /** @brief Read-only access to latitudes. */
    const std::vector<mfem::real_t>& Lats() const { return _lats; }

    /** @brief Longitude at index @p i. */
    mfem::real_t LonAt(int i) const { return _lons[i]; }
    /** @brief Latitude at index @p j. */
    mfem::real_t LatAt(int j) const { return _lats[j]; }

    /**
     * @brief Convert 2D indices (i,j) to a flat array index.
     *
     * @param i Longitude index in [0, NLon()).
     * @param j Latitude index in [0, NLat()).
     * @return Flattened index for row major layout.
     */
    size_t Idx(int i, int j) const;

    /**
     * @brief Evaluate a field value at the North Pole from field values at the highest latitude.
     *
     * @param field Flattened field values of size NLon()*NLat().
     * @return A interpolated value at lat = +90°.
     */
    mfem::real_t NorthPole(const std::vector<mfem::real_t>& field) const;

    /**
     * @brief Evaluate a field value at the South Pole from field values at the lowest latitude.
     *
     * @param field Flattened field values of size NLon()*NLat().
     * @return A interpolated value at lat = -90°.
     */
    mfem::real_t SouthPole(const std::vector<mfem::real_t>& field) const;

    /**
     * @brief Bilinear interpolation on the lon-lat grid.
     *
     * @param field Flattened field values of size NLon()*NLat().
     * @param lon Query longitude (degrees).
     * @param lat Query latitude (degrees).
     * @return Interpolated value at (lon, lat).
     */
    mfem::real_t Bilerp(const std::vector<mfem::real_t>& field, mfem::real_t lon, mfem::real_t lat) const;

private:
    std::vector<mfem::real_t> _lons, _lats; ///< Coordinate arrays (degrees).
    int _nlon = 0, _nlat = 0;               ///< Grid dimensions.
};

/**
 * @brief PREM (Preliminary Reference Earth Model) radial profiles and properties.
 *
 * Loads a PREM-like text file, skipping lines until the first line starting with '0.'.
 * Stores boundary radii (dimensional and non-dimensional) and optional property lists.
 */
class PREMModel {
public:
    /**
     * @brief Construct and load PREM data.
     *
     * @param fileName Path to a PREM-formatted text file.
     * @param Rref Reference radius used for non-dimensionalization.
     * @param buffer_ratio Radio of the buffer layer depth over the reference radius.
     * @param ignored_layers Number of outermost layers to ignore (default = 0).
     */
    PREMModel(const std::string& fileName,
              mfem::real_t Rref,
              mfem::real_t buffer_ratio,
              int ignored_layers = 0);

    /**
     * @brief Access non-dimensional radii.
     * @return Reference to radii divided by @ref _Rref.
     */
    std::vector<mfem::real_t>& GetRadiiND();

    /**
     * @brief Access dimensional radii.
     * @return Reference to radii in the same units as @ref _Rref.
     */
    std::vector<mfem::real_t>& GetRadii();

    ~PREMModel() = default;

private:
    mfem::real_t _Rref;           ///< Reference radius for non-dimensionalization.
    mfem::real_t _buffer_ratio;   ///< Radio of the buffer layer depth over the reference radius.
    int          _ignored_layers; ///< Number of outermost layers ignored on import.

    std::vector<mfem::real_t> radii;      ///< Dimensional layer boundary radii.
    std::vector<mfem::real_t> radii_nd;   ///< Non-dimensional radii (r / @ref _Rref).
    std::vector<mfem::real_t> density_list, pWave_list, sWave_list, bulkM_list, shearM_list; ///< Optional property tables.
};

/**
 * @brief Gridded scalar surface (e.g., topography) sampled on a lon-lat grid.
 *
 * Supports loading from XYZ (lon lat value) files, interpolation, arithmetic, and basic stats.
 */
class Topography {
public:
    Topography() = default;
    Topography(const Topography&) = default;
    Topography(Topography&&) noexcept = default;
    Topography& operator=(const Topography&) = default;
    Topography& operator=(Topography&&) noexcept = default;

    /**
     * @brief Construct from a XYZ file.
     *
     * @param xyzFile Path to a file with lines: "lon lat value" (degrees for lon/lat).
     * @param Rref Length scale for non-dimensionalisation (default 1.0).
     */
    Topography(const std::string& xyzFile, mfem::real_t Rref = 1.0);

    /**
     * @brief In-place pointwise addition.
     * @param other Another topography.
     * @return Reference to *this.
     *
     * @pre Grids can be on different coordinates but only the coordinates of *this are inherited.
     */
    Topography& operator+=(const Topography& other);

    /**
     * @brief Pointwise addition of two topographies.
     * @param A Left operand.
     * @param B Right operand.
     * @return A new Topography equal to A + B.
     *
     * @pre @pre Grids can be on different coordinates but only the coordinates of the left are inherited.
     */
    friend Topography operator+(const Topography& A, const Topography& B);

    /**
     * @brief Interpolate the surface value at (lon, lat).
     * @param lon Longitude in degrees.
     * @param lat Latitude in degrees.
     * @return Interpolated value.
     */
    mfem::real_t Interp(mfem::real_t lon, mfem::real_t lat) const;

    /** @brief Number of longitude samples. */
    int NLon() const;
    /** @brief Number of latitude samples. */
    int NLat() const;
    /** @brief Read-only longitudes. */
    const std::vector<mfem::real_t>& Lons() const;
    /** @brief Read-only latitudes. */
    const std::vector<mfem::real_t>& Lats() const;
    /** @brief Read-only flattened data array of size NLon()*NLat(). */
    const std::vector<mfem::real_t>& Data() const;

    /** @brief Longitude at index i. */
    mfem::real_t LonAt(int i) const;
    /** @brief Latitude at index j. */
    mfem::real_t LatAt(int j) const;

    /**
     * @brief Arithmetic mean of all grid values.
     * @return The mean value over @ref _data.
     */
    mfem::real_t Mean() const;

private:
    LonLatField _grid;              ///< Underlying lon-lat grid.
    mfem::real_t _Rref = 1.0;       ///< Length scale for non-dimensionalisation.
    std::vector<mfem::real_t> _data;///< Flattened data (size NLon()*NLat()).

    /**
     * @brief Construct directly from vectors.
     *
     * @param lons Longitudes (degrees).
     * @param lats Latitudes (degrees).
     * @param Rref Length scale.
     * @param data Flattened data of size lons.size() * lats.size().
     */
    Topography(std::vector<mfem::real_t> lons, std::vector<mfem::real_t> lats, mfem::real_t Rref, std::vector<mfem::real_t> data);

    /**
     * @brief Load XYZ into separate arrays.
     *
     * @param file Path to file.
     * @param L Output longitudes (degrees).
     * @param B Output latitudes (degrees).
     * @param V Output field values.
     * @return true on success, false on failure.
     */
    static bool LoadXYZ(const std::string& file,
                        std::vector<mfem::real_t>& L, std::vector<mfem::real_t>& B, std::vector<mfem::real_t>& V);

    /**
     * @brief Build a structured grid and data from scattered XYZ vectors.
     *
     * @param L Longitudes (degrees).
     * @param B Latitudes (degrees).
     * @param V Values aligned with (L,B).
     */
    void BuildGrid(const std::vector<mfem::real_t>& L,
                   const std::vector<mfem::real_t>& B,
                   const std::vector<mfem::real_t>& V);
};

/**
 * @brief Abstract interface for a radial surface r = R(lon, lat).
 */
class RadialSurface {
public:
    virtual ~RadialSurface() = default;
    /**
     * @brief Evaluate the radius at a given direction.
     *
     * @param lon Longitude (degrees).
     * @param lat Latitude (degrees).
     * @return Radius value.
     */
    virtual mfem::real_t RadiusAt(mfem::real_t lon, mfem::real_t lat) const = 0;
    //virtual mfem::real_t MeanRadius() const = 0;
};

/**
 * @brief Radial surface defined by a sampled field on a lon-lat grid.
 */
class FieldRadialSurface final : public RadialSurface {
public:
    /**
     * @brief Construct from a grid and corresponding radius field.
     *
     * @param grid Underlying lon-lat grid.
     * @param r_field Flattened radius field of size grid.NLon() * grid.NLat().
     */
    FieldRadialSurface(const LonLatField& grid, const std::vector<mfem::real_t>& r_field);
    /** @copydoc RadialSurface::RadiusAt */
    mfem::real_t RadiusAt(mfem::real_t lon, mfem::real_t lat) const override;
private:
    const LonLatField& _grid;                    ///< Reference to the sampling grid.
    const std::vector<mfem::real_t>& _r_field;   ///< Reference to the sampled radial field.
};

/**
 * @brief Spherical (constant-radius) surface r = const.
 */
class SpheroidalRadialSurface final : public RadialSurface {
public:
    /**
     * @brief Construct with constant radius.
     * @param r Sphere radius.
     */
    explicit SpheroidalRadialSurface(mfem::real_t r);
    /** @copydoc RadialSurface::RadiusAt */
    mfem::real_t RadiusAt(mfem::real_t, mfem::real_t) const override;
private:
    mfem::real_t _r; ///< Constant radius.
};

/**
 * @brief Triaxial ellipsoidal surface.
 *
 * Parameterized by semi-axes (a,b,c); returns the distance to the ellipsoid in the
 * (lon,lat) direction.
 */
class EllipsoidalRadialSurface final : public RadialSurface {
public:
    /**
     * @brief Construct an ellipsoidal surface.
     * @param a Semi-axis along x.
     * @param b Semi-axis along y.
     * @param c Semi-axis along z.
     */
    EllipsoidalRadialSurface(mfem::real_t a, mfem::real_t b, mfem::real_t c);
    /** @copydoc RadialSurface::RadiusAt */
    mfem::real_t RadiusAt(mfem::real_t lon, mfem::real_t lat) const override;
private:
    mfem::real_t _a, _b, _c; ///< Ellipsoid semi-axes.
};

//RadialMapping 
/**
 * @brief Base class for radial displacement mappings driven by a set of topographies.
 */
class RadialMapping {
public:
    /**
     * @brief Construct with a collection of topographies.
     *
     * @param topo Vector of pointers to topography fields (ownership not taken).
     * @param topo_exag Multiplicative exaggeration applied to topography when used (default 1.0).
     */
    RadialMapping(const std::vector<const Topography*>& topo, mfem::real_t topo_exag = 1.0);
    virtual ~RadialMapping() = default;

    /**
     * @brief Radial displacement at (r, lon, lat).
     *
     * @param r   Query radius (same units as topography reference).
     * @param lon Longitude (degrees).
     * @param lat Latitude (degrees).
     * @return Radial displacement (positive outward).
     */
    virtual mfem::real_t Displacement(mfem::real_t r, mfem::real_t lon, mfem::real_t lat) const = 0;

protected:
    /**
     * @brief Interpolate topography field @p i at (lon, lat) with exaggeration applied.
     *
     * @param i Index into @ref _topo.
     * @param lon Longitude (degrees).
     * @param lat Latitude (degrees).
     * @return Interpolated (and possibly scaled) topography value.
     *
     * @pre 0 <= i < _topo.size()
     */
    mfem::real_t InterpTopo(std::size_t i, mfem::real_t lon, mfem::real_t lat) const;

    const std::vector<const Topography*>& _topo; ///< External references to Topographies.
    mfem::real_t _topo_exag;                      ///< Exaggeration factor applied to topography.
};

/**
 * @brief Cubic band mapping with linear decay outside a [inner, outer] shell.
 *
 * Uses topographies referenced to two base radial surfaces, applying a cubic profile within the radial band
 * and a linear decay within depth @ref _decay outside.
 */
class CubicBandLinearDecay final : public RadialMapping {
public:
    /**
     * @brief Construct the mapping.
     *
     * @param topo      Topography fields used to drive the mapping.
     * @param base      Base radial surfaces.
     * @param decay     Layer depth for linear decay.
     * @param topo_exag Topography exaggeration factor (default 1.0).
     * @param iInner    Index into @p base for the inner surface (default 0).
     * @param iOuter    Index into @p base for the outer surface (default 1).
     */
    CubicBandLinearDecay(const std::vector<const Topography*>& topo,
                         const std::vector<const RadialSurface*>& base,
                         mfem::real_t decay,
                         mfem::real_t topo_exag = 1.0,
                         std::size_t iInner = 0,
                         std::size_t iOuter = 1);

    /**
     * @brief Radial displacement with cubic behavior in-band and linear decay out-of-band.
     *
     * @param r   Query radius.
     * @param lon Longitude (degrees).
     * @param lat Latitude (degrees).
     * @return Displacement (positive outward).
     */
    mfem::real_t Displacement(mfem::real_t r, mfem::real_t lon, mfem::real_t lat) const override;

private:
    const std::vector<const RadialSurface*>& _base; ///< Base radial surfaces (no ownership).
    mfem::real_t _decay = 0.0;                       ///< Decay layer depth.
    std::size_t _iInner = 0;                         ///< Index of inner base surface.
    std::size_t _iOuter = 1;                         ///< Index of outer base surface.
};

}// namespace mfemElasticity

