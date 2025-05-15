# mfemElasticity

This library contains extensions to the mfem library associated with the solution of elastic and viscoelastic problems. The particular focus is on geophysical applications. 

# Installation

The ```mfem``` library must first be installed. This can be either a serial or parallel version. 


Configuration of a build is done with CMake. The following options can be set:

- ```MFEM_DIR``` - The location on the mfem installation. If not set, standard locations will be search. If multiple versions of ```mfem``` have been installed (e.g., serial and parallel version) it is best to set this explicitly. 
- ```USE_MPI``` - Default is ```OFF```. If set to ```ON``` the library will link to the ```MPI``` library. This is necessary to use a parallel version of ```mfem``` and the associated functionality. 
- ```MPI_C_COMPILER``` - Path to the MPI c compiler used to build the MFEM Library. Useful to set if differention MPI compilers exists on your system. 
- ```MPI_CXX_COMPILER``` - Path to the MPI c++ compiler used to build the MFEM Library. Useful to set if differention MPI compilers exists on your system. 
- ```BUILD_EXAMPLES``` - Default is ```OFF```. If set to ```ON``` the example codes will be configured for compilation. 
- ```BUILD_TESTS``` - Default is ```OFF```. If set to ```ON``` the test codes will be configured for compilation. 

Other standard CMAKE options can additionally be set (e.g., specifying debug or release builds).

If either ```BUILD_EXAMPLES``` or ```BUILD_TESTS``` is selected, data files will be copied into the build directory. 

Note that "in source builds" are disabled. 

An example configuration for a parallel build is:

```
cmake -S mfemElasticity -B mfemElasticityBuild \
      -DUSE_MPI=ON \
      -DMFEM_DIR=[path to mfem build]  \
      -DMPI_C_COMPILER=[path to mpi c compiler] \
      -DMPI_CXX_COMPILER=[path to mpi c++ compiler] \
      -DBUILD_EXAMPLES=ON \
      -DBUILD_TESTS=ON
```

# Contents of the library

## Solvers

### RigidBodySolver

A custom ```mfem::Solver``` is defined for the solution of static linearised elastic boundary value problems subject to pure traction conditions. In such cases, a solution only exists if the forces and tractions apply no net torque or net force to the body. And the solution is only defined up to the addition of a linearised rigid body motion. 

The solver ```mfemElasticity::RigidBodySolver``` is formed by wrapping another solver for the problem. Its action on a right hand side vector proceeds by:

- Projecting orthogonally to the space of linearised rigidbody motions,
- Calling the underlying solver,
- Projecting the result orthogonally   to the space of linearised rigidbody motions. 

The first projection insures that the effective right hand side lies in the range of the linear equations and hence that a solution exists. The final projection removes any component in the kernel of the linear operator that might have been generated during the solution process. Doing so selects on particular solution to the problem. 

If the right hand side satisfies the existence conditions, then the first projection does nothing. If the existence condition is not met, then an exact solution to the equations is not produced. Instead, a minimum residual solution is found. 

## LinearFormsIntegrations

### DomainLFMatrixDeformationGradientIntegrator

An ```mfem::LinearFormIntegrator``` is defined the form:

$$
\mathbf{u} \mapsto \int_{\Omega} m_{ij} u_{i,j} \,\mathrm{d} x, 
$$

where $\mathbf{m}$ is a matrix-valued function and $\Omega$ is the domain. The matrix is specified as an ```mfem::MatrixCoefficient``` while the form acts on elements of a nodal finite element space that is formed from the product of scalar spaces. 

## BilinearFormIntegrators
