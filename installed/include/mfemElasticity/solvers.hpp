#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @brief Implementation of rigid body translations as a VectorCoefficient.
 *
 * This class represents a rigid body translation for use with MFEM's
 * VectorCoefficient. It defines a constant vector field where only
 * one component is non-zero (specifically, 1.0), representing a translation
 * along a specific axis.
 */
class RigidTranslation : public mfem::VectorCoefficient {
 private:
  int _component; /**< The component (spatial dimension) along which the
                       translation occurs (0 for x, 1 for y, 2 for z). */

 public:
  /**
   * @brief Constructor for the RigidTranslation class.
   *
   * To form a class instance, the spatial dimension and the component
   * of the translation are specified.
   *
   * @param dimension The spatial dimension of the problem (e.g., 2 for 2D, 3
   * for 3D).
   * @param component The index of the component (0, 1, or 2) along which the
   * translation is applied. Must be less than `dimension`.
   */
  RigidTranslation(int dimension, int component);

  /**
   * @brief Resets the component value for the translation.
   * @param component The new component index for the translation.
   */
  void SetComponent(int component);

  /**
   * @brief Overload of the Eval method for VectorCoefficients.
   *
   * This method evaluates the rigid translation vector at a given point.
   * The resulting vector will have a value of 1.0 in the specified `_component`
   * and 0.0 in all other components.
   *
   * @param V The output vector where the evaluated translation will be stored.
   * @param T The element transformation object.
   * @param ip The integration point where the coefficient is evaluated.
   */
  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override;
};

/**
 * @brief Implementation of rigid body rotations as a VectorCoefficient.
 *
 * This class represents a rigid body rotation for use with MFEM's
 * VectorCoefficient. It defines a vector field corresponding to a
 * rotation about a specific axis.
 *
 * @note If the spatial dimension is 3, then all three components (x, y, z)
 * of the rotation can be defined. If the spatial dimension is 2, only
 * the 2 (i.e. z) component of the rotation is defined (rotation in the XY
 * plane).
 */
class RigidRotation : public mfem::VectorCoefficient {
 private:
  int _component; /**< The component representing the axis of rotation.
                       In 2D, only 2 (z-axis) is valid. In 3D, 0 (x-axis),
                       1 (y-axis), or 2 (z-axis) are valid. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector
      _x; /**< Internal buffer for the transformed spatial coordinates. */
#endif

 public:
  /**
   * @brief Constructor for the RigidRotation class.
   *
   * To form a class instance, the spatial dimension and the component
   * of the rotation axis are specified.
   *
   * @param dimension The spatial dimension of the problem (e.g., 2 for 2D, 3
   * for 3D).
   * @param component The index of the rotation axis (0 for x, 1 for y, 2 for
   * z). In 2D, this must be 2 (z-axis rotation).
   */
  RigidRotation(int dimension, int component);

  /**
   * @brief Resets the component value for the rotation axis.
   * @param component The new component index for the rotation axis.
   */
  void SetComponent(int component);

  /**
   * @brief Overload of the Eval method for VectorCoefficients.
   *
   * This method evaluates the rigid rotation vector at a given point `ip`.
   * The rotation vector is computed based on the spatial coordinates and the
   * specified rotation axis.
   *
   * @param V The output vector where the evaluated rotation will be stored.
   * @param T The element transformation object.
   * @param ip The integration point where the coefficient is evaluated.
   */
  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override;
};

/**
 * @brief Linear solver suitable for static elastic problems with traction
 * boundary conditions.
 *
 * This class wraps another MFEM solver and pre- and post-multiplies
 * that solver's action with a projection orthogonal to the linearized rigid
 * body motions. This is crucial for solving problems where the stiffness matrix
 * has a null space corresponding to rigid body motions (e.g., free-floating
 * bodies under traction).
 *
 * This solver is suitable only for 2 and 3 dimensional problems.
 */
class RigidBodySolver : public mfem::Solver {
 private:
  mfem::FiniteElementSpace
      *_fes; /**< Pointer to the finite element space for displacements. */
  std::vector<std::unique_ptr<mfem::Vector>>
      _u; /**< A collection of orthogonalized rigid body modes (vectors). */
  mfem::Solver *_solver = nullptr; /**< Pointer to the underlying solver. */
  mutable mfem::Vector
      _b; /**< Internal buffer for the projected right-hand side vector. */
  const bool
      _parallel; /**< Flag indicating if the solver is running in parallel. */

#ifdef MFEM_USE_MPI
  mfem::ParFiniteElementSpace *_pfes; /**< Pointer to the parallel finite
                                         element space (if MPI is enabled). */
  MPI_Comm _comm; /**< The MPI communicator (if MPI is enabled). */
#endif

  /**
   * @brief Local implementation of the Euclidean dot product of two vectors.
   *
   * This method correctly computes the dot product for both serial and parallel
   * codes.
   *
   * @param x The first vector.
   * @param y The second vector.
   * @return The Euclidean dot product of x and y.
   */
  mfem::real_t Dot(const mfem::Vector &x, const mfem::Vector &y) const;

  /**
   * @brief Returns the Euclidean norm of a vector.
   *
   * @param x The vector for which to compute the norm.
   * @return The Euclidean norm of the vector x.
   */
  mfem::real_t Norm(const mfem::Vector &x) const;

  /**
   * @brief Sets up the rigid body fields.
   *
   * This method generates the basis vectors for rigid body translations
   * and rotations based on the spatial dimension of the problem and
   * projects them onto the finite element space. These vectors are then
   * orthogonalized using Gram-Schmidt.
   */
  void SetRigidBodyFields();

  /**
   * @brief Performs modified Gram-Schmidt orthogonalization on the linearized
   * rigid body modes.
   *
   * This ensures that the basis vectors for the null space are orthonormal.
   */
  void GramSchmidt();

  /**
   * @brief Returns the dimension of the space of linearized rigid body motions.
   *
   * For 2D, this is 3 (2 translations + 1 rotation).
   * For 3D, this is 6 (3 translations + 3 rotations).
   *
   * @return The dimension of the null space.
   */
  int GetNullDim() const;

  /**
   * @brief Takes in a vector x and returns in y its projection orthogonal to
   * the linearized rigid body modes.
   *
   * This operation removes any rigid body components from the given vector.
   *
   * @param x The input vector.
   * @param y The output vector, which is the projection of x orthogonal to the
   * rigid body modes.
   */
  void ProjectOrthogonalToRigidBody(const mfem::Vector &x,
                                    mfem::Vector &y) const;

 public:
  /**
   * @brief Constructor for the RigidBodySolver class for serial execution.
   *
   * To construct a class instance, the finite element space on
   * which the displacements are defined must be provided.
   *
   * @param fes Pointer to the `mfem::FiniteElementSpace`.
   */
  RigidBodySolver(mfem::FiniteElementSpace *fes);

#ifdef MFEM_USE_MPI
  /**
   * @brief Constructor for the RigidBodySolver class for parallel execution.
   *
   * To construct a class instance, the MPI communicator and
   * finite element space on which the displacements are defined
   * must be provided.
   *
   * @param comm The MPI communicator.
   * @param fes Pointer to the `mfem::ParFiniteElementSpace`.
   */
  RigidBodySolver(MPI_Comm comm, mfem::ParFiniteElementSpace *fes);
#endif

  /**
   * @brief Sets the underlying MFEM solver to be used.
   *
   * This solver will be wrapped by the `RigidBodySolver`.
   *
   * @param solver A reference to the `mfem::Solver` instance.
   */
  void SetSolver(mfem::Solver &solver);

  /**
   * @brief Sets the operator for the underlying solver.
   *
   * The `SetSolver()` method must be called before calling this routine.
   * If the operator is already set within the underlying solver, then this
   * method need not be called.
   *
   * @param op A constant reference to the `mfem::Operator` to be set.
   */
  void SetOperator(const mfem::Operator &op) override;

  /**
   * @brief Implementation of the Mult method, which performs the solver's
   * action.
   *
   * The action of the underlying solver is wrapped with projections orthogonal
   * to the rigid body modes. This ensures that the solution `x` is also
   * orthogonal to the rigid body modes, and the right-hand side `b` is
   * similarly projected before being passed to the underlying solver.
   *
   * @param b The right-hand side vector.
   * @param x The solution vector.
   */
  void Mult(const mfem::Vector &b, mfem::Vector &x) const override;
};

}  // namespace mfemElasticity