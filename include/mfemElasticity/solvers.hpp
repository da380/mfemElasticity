#pragma once

#include <memory>

#include "mfem.hpp"

namespace mfemElasticity {

/*
Implementation of rigid body translations as a VectorCoefficient.
*/
class RigidTranslation : public mfem::VectorCoefficient {
 private:
  int _component;

 public:
  /*
    To form a class instance, the spatial dimension and component
    are specified.
  */
  RigidTranslation(int dimension, int component);

  /*
    Reset the component value.
  */
  void SetComponent(int component);

  /*
    Overload of the Eval method for VectorCoefficients.
  */
  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override;
};

/*
Implementation of rigid body rotations as a VectorCoefficient.

Note that if the spatial dimension is 3, then only the 2 (i.e. z)
component of the rotation is define.
*/
class RigidRotation : public mfem::VectorCoefficient {
 private:
  int _component;
  mfem::Vector _x;

 public:
  /*
    To form a class instance, the spatial dimension and component
    are specified.
  */
  RigidRotation(int dimension, int component);

  /*
    Reset the component value.
  */
  void SetComponent(int component);

  /*
    Overload of the Eval method for VectorCoefficients.
  */
  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override;
};

/*
Linear solver suitable for static elastic problems
with traction boundary conditions. This class is used to
wrap another solver  and t simply pre- and post-multiplies
that solvers action with a projection orthogonal to the
linearisd rigid body motions.

This solver is suitable only for 2 and 3 dimensional problems.
*/
class RigidBodySolver : public mfem::Solver {
 private:
  mfem::FiniteElementSpace *_fes;
  std::vector<std::unique_ptr<mfem::Vector>> _u;
  mfem::Solver *_solver = nullptr;
  mutable mfem::Vector _b;
  const bool _parallel;

#ifdef MFEM_USE_MPI
  mfem::ParFiniteElementSpace *_pfes;
  MPI_Comm _comm;
#endif

  /*
    Local implementation of the Euclidean dot product of two
    vectors that applies for both serial and parallel codes.
  */
  mfem::real_t Dot(const mfem::Vector &x, const mfem::Vector &y) const;

  /*
    Returns the Euclidean norm of a vector.
  */
  mfem::real_t Norm(const mfem::Vector &x) const;

  /*
  Sets up the rigid body fields.
  */
  void SetRigidBodyFields();

  /*
    Performs modified Gram-Schmidt orthogonalisation
    on the linearised rigid body modes.
  */
  void GramSchmidt();

  /*
    Returns the dimension of the space of linearised
    rigid body motions.
  */
  int GetNullDim() const;

  /*
    Takes in a vector x and returns in y its projection
    orthogonal to the linearised rigid body modes.
  */
  void ProjectOrthogonalToRigidBody(const mfem::Vector &x,
                                    mfem::Vector &y) const;

 public:
  /*
    To construct a class instance the finite element space on
    which the displacements are defined must be provided.

    This constructor can be used only within serial codes.
  */
  RigidBodySolver(mfem::FiniteElementSpace *fes);

#ifdef MFEM_USE_MPI
  /*
    To construct a class instance the MPI communicator and
    finite element space on which the displacements are defined
    must be provided.

    This constructor can be used only within parallel codes.
  */
  RigidBodySolver(MPI_Comm comm, mfem::ParFiniteElementSpace *fes);
#endif

  /*
    Set the underlying solver.
  */
  void SetSolver(mfem::Solver &solver);

  /*
    Set the operator. The solver must be set before calling
    this routine. If the operator is already set within the
    solver, then this method need not be called.
  */
  void SetOperator(const mfem::Operator &op) override;

  /*
    Implementation of the Mult method. The action of the
    underlying solver is wrapped with projections orhtogonal
    to the rigid body modes.
  */
  void Mult(const mfem::Vector &b, mfem::Vector &x) const override;
};

}  // namespace mfemElasticity