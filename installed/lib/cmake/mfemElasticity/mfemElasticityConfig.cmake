
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was Config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

####################################################################################

include(CMakeFindDependencyMacro)

set(USE_MPI ON)

set(MFEM_DIR )
find_dependency(mfem NAMES mfem MFEM HINTS ${MFEM_DIR})
message(STATUS "Found mfem config in: ${mfem_DIR} (version ${MFEM_VERSION})")


if(USE_MPI)
    enable_language(C)
    set(MPIEXEC_EXECUTABLE /home/sssou/source/petsc/arch-opt/bin/mpiexec)    
    find_package(MPI REQUIRED)
    if (NOT CMAKE_CXX_COMPILER AND /home/sssou/source/petsc/arch-opt/bin/mpicxx)  
        set(CMAKE_CXX_COMPILER "/home/sssou/source/petsc/arch-opt/bin/mpicxx")      
    endif()
endif()

include ( "${CMAKE_CURRENT_LIST_DIR}/mfemElasticityTargets.cmake" )

