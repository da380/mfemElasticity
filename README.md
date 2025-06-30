# mfemElasticity

This library contains extensions to the mfem library associated with the solution of elastic and viscoelastic problems. 

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

## ```mfem::Solvers```

### RigidBodySolver

An instance of the ```mfem::Solver``` class that wraps another solver, and orthogonally projects out rigid body motions. Used in the solution of static or quasi-static problems with traction boundary conditions. See ```ex1.cpp``` and ```ex1p.cpp``` for an example of its usage. 

## ```mfem::LinearFormIntegrator``` and   ```mfem::BilinearFormIntegrator```

A number of classes derived from ```mfem::LinearFormIntegrator``` and ```mfem::BilinearFormIntegrator``` are defined. In each case, vector or matrix fields are defined on finite-element spaces formed by appropriate tensor-products of a scalar-finite element space. 

### DomainLFDeformationGradientIntegrator

The ```mfem::LinearFormIntegrator``` associated with

$$
u \mapsto \int_{\Omega} m_{ij} u_{i,j} \,\mathrm{d} x, 
$$

for a vector-field $u$ and a matrix-field $m_{ij}$. 


###  DomainVectorScalarIntegrator

The ```mfem::BilinearFormIntegrator``` associated with 

$$
(v,u) \mapsto \int_{\Omega} q_{i} v_{i} u \,\mathrm{d} x, 
$$

where $\Omega$ is a domain, $q$ a vector-coefficient, $v$ a vector test function, and $u$ a scalar trial function.

### DomainVectorGradScalarIntegrator

The ```mfem::BilinearFormIntegrator``` associated with

$$
(v, u) \mapsto \int_{\Omega} v_{i} q_{ij} u_{,j} \,\mathrm{d} x, 
$$

where $\Omega$ is a domain, $q$ a matrix-coefficient, $v$ a vector test function, and $u$ a scalar trial function. The matrix coefficient can be input as:

- A ```Coefficient```, in which case $q$ is proportional to the identity matrix;
- A ```VectorCoefficient```, in which case $q$ is a diagonal matrix;
- A ```MatrixCoefficient``` which is the general case. 

### DomainDivVectorScalarIntegrator

The ```mfem::BilinearFormIntegrator``` associated with 

$$
(v,u) \mapsto \int_{\Omega} q \,v_{i,i} u \,\mathrm{d} x, 
$$

where $\Omega$ is a domain, $v$ a vector test function, and $u$ a 
scalar trial function. 

### DomainDivVectorDivVectorIntegrator

The ```mfem::BilinearFormIntegrator``` associated with 

$$
(v,u) \mapsto \int_{\Omega} q \,v_{i,i} u_{j,j} \,\mathrm{d} x, 
$$

where $\Omega$ is a domain, $v$ a vector test function, and $u$ a vector trial function. 

### DomainVectorGradVectorIntegrator

The ```mfem::BilinearFormIntegrator``` associated with 

$$
(v, u) \mapsto \int_{\Omega} q v_{i} (w_{j} u_{j})_{,i} \mathrm{d} x, 
$$

where $q$ is a scalar coefficient, $v$ a vector test function, 
$w$ a vector coefficient, and $u$ a vector trial function. 

### DomainVectorDivVectorIntegrator

The ```mfem::BinlinearFormIntegrator``` associated with 

$$
(v,u) \mapsto \int_{\Omega} q_{i} v_{i} u_{j,j}\,\mathrm{d }x,
$$

where $\Omega$ is a domain, $q$ a vector coefficient, $v$ a vector 
test function, and $u$ a vector trial function. 


### DomainMatrixDeformationGradientIntegrator

The ```mfem::BilinearFormIntegrator``` associated with 

$$
(v,u) \mapsto \int_{\Omega} q\, v_{ij} u_{i,j} \,\mathrm{d} x, 
$$

where $q$ is a scalar coefficient, $v$ a matrix test function, and $u$
a vector trial function. The matrix field is implemented as a $n^{2}$-dimensional vector ```mfem::GridFunction``` with the matrix's components stored in column-major order. In 2D, for example, this ordering corresponds to:

$$
\left(\begin{array}{cc}
v_{00} & v_{01} \\ v_{10} & v_{11}
\end{array}\right) \mapsto \left(
\begin{array}{c}
v_{00} \\ v_{10} \\ v_{01} \\ v_{11}
\end{array}
\right).
$$

### DomainSymmetricMatrixStrainIntegrator

The ```mfem::BilinearFormIntegrator``` associated with 

$$
(v,u) \mapsto \int_{\Omega} q\, v_{ij} u_{i,j} \,\mathrm{d} x, 
$$

where $q$ is a scalar coefficient, $v$ a symmetric matrix test function, and $u$
a vector trial function. The matrix field is implemented as a $\frac{1}{2}n(n+1)$-dimensional vector ```mfem::GridFunction``` with the matrix's components from the lower-triangle stored in column-major order.
In 2D, for example, this ordering corresponds to:

$$
\left(\begin{array}{cc}
v_{00} & v_{01} \\ v_{01} & v_{11}
\end{array}\right) \mapsto \left(
\begin{array}{c}
v_{00} \\ v_{01}  \\ v_{11}
\end{array}
\right).
$$ 

### DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator

The ```mfem::BilinearFormIntegrator``` associated with 

$$
(v,u) \mapsto \int_{\Omega} q\, v_{ij} u_{i,j} \,\mathrm{d} x, 
$$

where $q$ is a scalar coefficient, $v$ a symmetric matrix test function, and $u$
a vector trial function. The trace-freee and symmetric matrix field is implemented as a $\frac{1}{2}n(n+1)-1$-dimensional vector ```mfem::GridFunction``` with the matrix's components from the lower-triangle stored in column-major order but with the final element removed.  In 2D, for example, this ordering corresponds to:

$$
\left(\begin{array}{cc}
v_{00} & v_{01} \\ v_{01} & -v_{00}
\end{array}\right) \mapsto \left(
\begin{array}{c}
v_{00} \\ v_{01}
\end{array}
\right).
$$ 

## ```mfem::DiscreteInterpolator```

A number of classes derived from ```mfem::DiscreteInterpolator``` are defined. Matrix fields are implemented using vector 
instances of ```mfem::Gridfunction``` with the appropriate dimension and using the ordering conventions discussed above. 

### DeformationGradientInterpolator
The ```mfem::DiscreteInterpolator``` associated with 

$$
u_{i} \mapsto u_{i,j}, 
$$
for a vector field. The result is a matrix field. 

### StrainInterpolator
The ```mfem::DiscreteInterpolator``` associated with 

$$
u_{i} \mapsto \frac{1}{2}(u_{i,j} + u_{j,i}), 
$$
for a vector field. The result is a symmetric matrix field. 

### DeviatoricStrainInterpolator 

The ```mfem::DiscreteInterpolator``` associated with 

$$
u_{i} \mapsto \frac{1}{2}(u_{i,j} + u_{j,i}) - \frac{1}{n}u_{k,k}\delta_{ij}
$$
for a vector field. The result is a trace-free symmetric matrix field. 