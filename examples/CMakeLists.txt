# The examples. transformed_diffusion and elastogravity_layered are one source
# for both builds: parallel programs when MFEM has MPI, serial ones otherwise.
# The examples that differ in parallel have a counterpart (suffix _p), added
# below when USE_MPI is on. The examples read their meshes from <build>/data,
# which the top-level CMakeLists.txt fills (CopyData and the meshes/ scripts).

set(SERIAL_EXAMPLES
    poisson_dtn                 # Poisson on the whole space: Neumann / DtN / multipole
    transformed_diffusion       # pull-back Laplacian on a mapped domain
    submesh_injection           # tour of SubMeshDofInjection
    coupled_poisson             # coupled Poisson pair across mesh / submesh
    elastogravity_layered       # LinearQuasiStaticSelfGravitatingProblem: uniform, fluid core (+ inner core)
    quasi_static_elasticity     # quasi_static_problem.hpp driver
    viscoelasticity             # ViscoelasticOperator driver
    viscoelastic_loading        # GIA-style layered load/rebound driver
    viscoelastic_schemes        # cost/accuracy table of the time integrators
    self_gravitating_relaxation # Maxwell mantle + fluid core under a Heaviside load
    love_numbers                # load and tidal Love numbers by degree, one solve each
    anisotropic_elasticity      # ElasticTensorIntegrator, radial TI
)

set(PARALLEL_EXAMPLES
    poisson_dtn_p
    submesh_injection_p
    coupled_poisson_p
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

