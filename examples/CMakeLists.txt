# The examples: one source per example, for both builds — a parallel
# program when MFEM has MPI, a serial one otherwise, with the genuine
# serial/parallel differences marked by #ifdef MFEM_USE_MPI blocks in the
# sources. The examples read their meshes from <build>/data, which the
# top-level CMakeLists.txt fills (CopyData and the meshes/ scripts).

set(EXAMPLES
    poisson_dtn                 # Poisson on the whole space: Neumann / DtN / multipole
    transformed_diffusion       # pull-back Laplacian on a mapped domain
    submesh_injection           # tour of SubMeshDofInjection
    coupled_poisson             # coupled Poisson pair across mesh / submesh
    elastogravity_layered       # LinearQuasiStaticSelfGravitatingProblem: uniform, fluid core (+ inner core)
    quasi_static_elasticity     # quasi_static_problem.hpp driver
    gauged_fluid_cavity         # gauged fluid region, purely elastic (SetGaugedFluid)
    sliding_fluid_ellipse       # sliding interface (displacement discontinuity) on an elliptical body
    equilibrium_stress          # AW10 minimum-norm / minimum-deviatoric equilibrium stress generators
    prestress_loading           # does the deviatoric pre-stress matter for loading? (ellipse sweep)
    referential_elastogravity   # general referential problem vs the Eulerian one (self-benchmark)
    viscoelasticity             # ViscoelasticOperator driver
    viscoelastic_loading        # GIA-style layered load/rebound driver
    viscoelastic_schemes        # cost/accuracy table of the time integrators
    self_gravitating_relaxation # Maxwell mantle + fluid core under a Heaviside load
    self_gravitating_solvers    # one problem, every linear-solver architecture: cost/agreement table
    love_numbers                # load and tidal Love numbers by degree, one solve each
    anisotropic_elasticity      # ElasticTensorIntegrator, radial TI
)

foreach(ex IN LISTS EXAMPLES)
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} mfemElasticity)
endforeach()

