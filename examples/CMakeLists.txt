# Serial examples. Each parallel counterpart (suffix _p) is added below when
# USE_MPI is on. Meshes referenced by the examples live in ../data and are
# copied to the build tree by the CopyData target.

set(SERIAL_EXAMPLES
    poisson_dtn                 # Poisson on the whole space: Neumann / DtN / multipole
    transformed_diffusion       # pull-back Laplacian on a mapped domain
    submesh_injection           # tour of SubMeshDofInjection
    coupled_poisson             # coupled Poisson pair across mesh / submesh
    elastogravity               # self-gravitating elastic body, block MINRES
    elastogravity_two_layer     # ... liquid core + solid mantle
    elastogravity_three_layer   # ... solid inner core + liquid outer core + mantle
    quasi_static_elasticity     # elastic_problem.hpp driver
    viscoelasticity             # ViscoelasticOperator driver
    self_gravitating_elasticity # SelfGravitatingElasticProblem driver
    anisotropic_elasticity      # ElasticTensorIntegrator, radial TI
)

set(PARALLEL_EXAMPLES
    poisson_dtn_p
    transformed_diffusion_p
    submesh_injection_p
    coupled_poisson_p
    elastogravity_p
    elastogravity_two_layer_p
    elastogravity_three_layer_p
    self_gravitating_elasticity_p
)

foreach(ex IN LISTS SERIAL_EXAMPLES)
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} mfemElasticity)
endforeach()

if(USE_MPI)
    foreach(ex IN LISTS PARALLEL_EXAMPLES)
        add_executable(${ex} ${ex}.cpp)
        target_link_libraries(${ex} mfemElasticity)
    endforeach()
endif()

