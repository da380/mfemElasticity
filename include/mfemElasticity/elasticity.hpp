#pragma once

#include <memory>
#include <regex>

#include "mesh.hpp"

namespace mfemElasticity {

/**
 * @brief Base class for quasi-static elasticity problems.
 *
 * This abstract class provides a consistent interface for formulating
 * quasi-static linear elastic problems. This allows any such problems
 * to be readily coupled to a viscoelastic model.
 */
class QuasiStaticElasticityProblem {
 private:
  /**
   * Pointer to finite element space used for the displacement field. This
   * is not owned by the class.
   */
  mfem::FiniteElementSpace* ufes_;

  /**
   * A unique pointer to a GridFunction for the displacement vector.
   */
  std::unique_ptr<mfem::GridFunction> u_;

  /**
   *A vector for the displacement vector.
   */
  mfem::Vector x_;

  /**
   *A vector for the RHS of the elasticity problem.
   */
  mfem::Vector y_;

 public:
  /**
   * Constructor given a pointer to the finite element space for the
   * displacement field.
   */
  QuasiStaticElasticityProblem(mfem::FiniteElementSpace* ufes) : ufes_{ufes} {}

  virtual ~QuasiStaticElasticityProblem() = default;

  /**
   * Pure virtual method that computes the solution of the quasi-static problem
   * at the given time and using the right hand side that has been set. The
   * solution is stored within the internal GridFunction for the displacement
   * vector.
   */
  virtual void Solve() = 0;

  /**
   * Pure virtual method that sets up the right hand side for the problem at the
   * set time.
   */
  virtual void SetRHS() = 0;

  /**
   * A method that increments the right hand side using an input
   * vector. This can be used within viscoelastic problems to add
   * in relaxation forces prior to solution of the linear syetem.
   */
  void IncrementRHS(mfem::Vector& y) {
    MFEM_ASSERT(y_.Size() == y.Size(),
                "Increment to the RHS has the wrong dimension");
    y_ += y;
  }
};

}  // namespace mfemElasticity