include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)


add_executable(TestDomainLFDeformationGradientIntegrator TestDomainLFDeformationGradientIntegrator.cpp)
target_link_libraries(TestDomainLFDeformationGradientIntegrator PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestDomainLFDeformationGradientIntegrator)

add_executable(TestInterpolators TestInterpolators.cpp)
target_link_libraries(TestInterpolators PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestInterpolators)

add_executable(TestDomainVectorScalarIntegrator TestDomainVectorScalarIntegrator.cpp)
target_link_libraries(TestDomainVectorScalarIntegrator PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestDomainVectorScalarIntegrator)

add_executable(TestDomainVectorGradScalarIntegrator TestDomainVectorGradScalarIntegrator.cpp)
target_link_libraries(TestDomainVectorGradScalarIntegrator PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestDomainVectorGradScalarIntegrator)

add_executable(TestDomainDivVectorScalarIntegrator TestDomainDivVectorScalarIntegrator.cpp)
target_link_libraries(TestDomainDivVectorScalarIntegrator PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestDomainDivVectorScalarIntegrator)

add_executable(TestDomainDivVectorDivVectorIntegrator TestDomainDivVectorDivVectorIntegrator.cpp)
target_link_libraries(TestDomainDivVectorDivVectorIntegrator PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestDomainDivVectorDivVectorIntegrator)

add_executable(TestDomainVectorGradVectorIntegrator TestDomainVectorGradVectorIntegrator.cpp)
target_link_libraries(TestDomainVectorGradVectorIntegrator PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestDomainVectorGradVectorIntegrator)

add_executable(TestDomainVectorDivVectorIntegrator TestDomainVectorDivVectorIntegrator.cpp)
target_link_libraries(TestDomainVectorDivVectorIntegrator PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestDomainVectorDivVectorIntegrator)

add_executable(TestDomainMatrixDeformationGradientIntegrator TestDomainMatrixDeformationGradientIntegrator.cpp)
target_link_libraries(TestDomainMatrixDeformationGradientIntegrator PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestDomainMatrixDeformationGradientIntegrator)

add_executable(TestDomainSymmetricMatrixStrainIntegrator TestDomainSymmetricMatrixStrainIntegrator.cpp)
target_link_libraries(TestDomainSymmetricMatrixStrainIntegrator PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestDomainSymmetricMatrixStrainIntegrator)


add_executable(TestDomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator TestDomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator.cpp)
target_link_libraries(TestDomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestDomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator)

add_executable(TestBoundaryNormalIntegrators TestBoundaryNormalIntegrators.cpp)
target_link_libraries(TestBoundaryNormalIntegrators PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestBoundaryNormalIntegrators)

add_executable(TestSubMeshDofInjection TestSubMeshDofInjection.cpp)
target_link_libraries(TestSubMeshDofInjection PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestSubMeshDofInjection)

if(USE_MPI)
    add_executable(TestSubMeshDofInjectionPar TestSubMeshDofInjectionPar.cpp)
    target_link_libraries(TestSubMeshDofInjectionPar PRIVATE mfemElasticity MPI::MPI_CXX)
    add_executable(TestSubMeshMixedBilinearFormPar TestSubMeshMixedBilinearFormPar.cpp)
    target_link_libraries(TestSubMeshMixedBilinearFormPar PRIVATE mfemElasticity MPI::MPI_CXX)
    add_executable(TestQuasiStaticProblemPar TestQuasiStaticProblemPar.cpp)
    target_link_libraries(TestQuasiStaticProblemPar PRIVATE mfemElasticity MPI::MPI_CXX)
    add_executable(TestViscoelasticPar TestViscoelasticPar.cpp)
    target_link_libraries(TestViscoelasticPar PRIVATE mfemElasticity MPI::MPI_CXX)
    add_executable(TestSelfGravitatingPar TestSelfGravitatingPar.cpp)
    target_link_libraries(TestSelfGravitatingPar PRIVATE mfemElasticity MPI::MPI_CXX)
    add_executable(TestBoundaryNormalIntegratorsPar TestBoundaryNormalIntegratorsPar.cpp)
    target_link_libraries(TestBoundaryNormalIntegratorsPar PRIVATE mfemElasticity MPI::MPI_CXX)
    add_executable(TestSelfGravitatingFluidPar TestSelfGravitatingFluidPar.cpp)
    target_link_libraries(TestSelfGravitatingFluidPar PRIVATE mfemElasticity MPI::MPI_CXX)
    add_executable(TestSolversPar TestSolversPar.cpp)
    target_link_libraries(TestSolversPar PRIVATE mfemElasticity MPI::MPI_CXX)
    add_executable(TestCompositeRheologyPar TestCompositeRheologyPar.cpp)
    target_link_libraries(TestCompositeRheologyPar PRIVATE mfemElasticity MPI::MPI_CXX)
    add_executable(TestSphericalHarmonicsPar TestSphericalHarmonicsPar.cpp)
    target_link_libraries(TestSphericalHarmonicsPar PRIVATE mfemElasticity MPI::MPI_CXX)
    foreach(np 1 2 4)
        add_test(NAME TestSubMeshDofInjectionPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSubMeshDofInjectionPar>)
        add_test(NAME TestSubMeshMixedBilinearFormPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSubMeshMixedBilinearFormPar>)
        add_test(NAME TestQuasiStaticProblemPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestQuasiStaticProblemPar>)
        add_test(NAME TestViscoelasticPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestViscoelasticPar>)
        add_test(NAME TestSelfGravitatingPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSelfGravitatingPar>)
        add_test(NAME TestBoundaryNormalIntegratorsPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestBoundaryNormalIntegratorsPar>)
        add_test(NAME TestSelfGravitatingFluidPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSelfGravitatingFluidPar>)
        add_test(NAME TestSolversPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSolversPar>)
        add_test(NAME TestCompositeRheologyPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestCompositeRheologyPar>)
        add_test(NAME TestSphericalHarmonicsPar_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                         $<TARGET_FILE:TestSphericalHarmonicsPar>)
    endforeach()
endif()


add_executable(TestSolvers TestSolvers.cpp)
target_link_libraries(TestSolvers PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestSolvers)

add_executable(TestRelaxationLaw TestRelaxationLaw.cpp)
target_link_libraries(TestRelaxationLaw PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestRelaxationLaw)

add_executable(TestSphericalHarmonics TestSphericalHarmonics.cpp)
target_link_libraries(TestSphericalHarmonics PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestSphericalHarmonics)

add_executable(TestCompositeRheology TestCompositeRheology.cpp)
target_link_libraries(TestCompositeRheology PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestCompositeRheology)

add_executable(TestRheology TestRheology.cpp)
target_link_libraries(TestRheology PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestRheology)

add_executable(TestQuasiStaticProblem TestQuasiStaticProblem.cpp)
target_link_libraries(TestQuasiStaticProblem PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestQuasiStaticProblem)

add_executable(TestElasticTensor TestElasticTensor.cpp)
target_link_libraries(TestElasticTensor PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestElasticTensor)

add_executable(TestElasticTensorIntegrator TestElasticTensorIntegrator.cpp)
target_link_libraries(TestElasticTensorIntegrator PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestElasticTensorIntegrator)

add_executable(TestViscoelastic TestViscoelastic.cpp)
target_link_libraries(TestViscoelastic PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestViscoelastic)

add_executable(TestSubMeshMixedBilinearForm TestSubMeshMixedBilinearForm.cpp)
target_link_libraries(TestSubMeshMixedBilinearForm PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestSubMeshMixedBilinearForm)

add_executable(TestTransformedDiffusionIntegrator TestTransformedDiffusionIntegrator.cpp)
target_link_libraries(TestTransformedDiffusionIntegrator PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestTransformedDiffusionIntegrator)

add_executable(TestSelfGravitating TestSelfGravitating.cpp)
target_link_libraries(TestSelfGravitating PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestSelfGravitating)

add_executable(TestSelfGravitatingFluid TestSelfGravitatingFluid.cpp)
target_link_libraries(TestSelfGravitatingFluid PRIVATE mfemElasticity gtest_main)
gtest_discover_tests(TestSelfGravitatingFluid)
