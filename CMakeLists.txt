cmake_minimum_required(VERSION 3.15)

project(
    mfemElasticity
    VERSION 0.0.1
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prohibit in-source builds
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are prohibited.")
endif ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Import MFEM. The following variables can be used to help CMake find MFEM:
#  * MFEM_DIR - absolute path to the MFEM build or install prefix.
#  * mfem_DIR - absolute path to where MFEMConfig.cmake is.
message(STATUS "Looking for mfem ...")
set(MFEM_DIR "" CACHE PATH "Path to the MFEM build or install prefix.")
if (MFEM_DIR)
    find_package(mfem REQUIRED NAMES MFEM HINTS "${MFEM_DIR}"
                "${mfem_DIR}/lib/cmake/mfem" NO_DEFAULT_PATH)
else()
    find_package(mfem CONFIG REQUIRED NAMES MFEM)
endif()
message(STATUS "Found mfem config in: ${mfem_DIR} (version ${MFEM_VERSION})")





# Add MPI is needed. 
option(USE_MPI "Use MPI library" OFF)
if(USE_MPI)
    find_package(MPI REQUIRED)
    if (NOT CMAKE_CXX_COMPILER AND MFEM_CXX_COMPILER)  
        set(CMAKE_CXX_COMPILER "${MFEM_CXX_COMPILER}")      
    endif()
endif()

# Set up the library 
add_library(${PROJECT_NAME} STATIC)

# Add in linked libraries.
if(USE_MPI)
    target_link_libraries(${PROJECT_NAME} PUBLIC mfem MPI::MPI_CXX)
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC mfem)
endif()


# Add in the source files. 
add_subdirectory(src)

# Optionally build examples. 
option(BUILD_EXAMPLES "Build the example programs" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Optionally build tests
option(BUILD_TESTS "Build the test programs" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES OR BUILD_TESTS)
    add_custom_target(CopyData        
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data ${CMAKE_BINARY_DIR}/data        
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            )
    add_dependencies(mfemElasticity CopyData)        
endif()

# Add standard install locations. 
include(GNUInstallDirs)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install your library (headers and library file)
install(TARGETS mfemElasticity
        EXPORT mfemElasticityTargets
        DESTINATION lib)
install(DIRECTORY include/
        DESTINATION include)


# Create the CMake package configuration file
install(EXPORT mfemElasticityTargets
        NAMESPACE mfemElasticity::
        FILE mfemElasticityTargets.cmake
        DESTINATION lib/cmake/mfemElasticity)

include(CMakePackageConfigHelpers)        


set(INCLUDE_INSTALL_DIRS ${PROJECT_BINARY_DIR})

set(LIB_INSTALL_DIR ${PROJECT_BINARY_DIR})
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/mfemElasticityConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/mfemElasticity"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR
  )

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/mfemElasticityConfigVersion.cmake"
  VERSION "${mfemElasticity_VERSION_MAJOR}.${mfemElasticity_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/mfemElasticityConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/mfemElasticityConfigVersion.cmake
  DESTINATION lib/cmake/mfemElasticity
  )

export(EXPORT mfemElasticityTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/mfemElasticityTargets.cmake"
)



