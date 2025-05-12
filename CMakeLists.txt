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


# Import MFEM. The following variables can be used to help CMake find MFEM:
#  * MFEM_DIR - absolute path to the MFEM build or install prefix.
#  * mfem_DIR - absolute path to where MFEMConfig.cmake is.
message(STATUS "Looking for mfem ...")
set(MFEM_DIR "" CACHE PATH "Path to the MFEM build or install prefix.")
if (MFEM_DIR)
   find_package(mfem REQUIRED NAMES MFEM HINTS "${MFEM_DIR}"
                "${mfem_DIR}/lib/cmake/mfem" NO_DEFAULT_PATH)
else()
   find_package(mfem REQUIRED NAMES MFEM)
endif()
message(STATUS "Found mfem config in: ${mfem_DIR} (version ${MFEM_VERSION})")

## Make sure consistent C++ compilers are used. 
if (NOT CMAKE_CXX_COMPILER AND MFEM_CXX_COMPILER)  
  set(CMAKE_CXX_COMPILER "${MFEM_CXX_COMPILER}")      
endif()

# Find some packages.
find_package(MPI REQUIRED)
#find_package(Boost)


# Set up the library 
add_library(${PROJECT_NAME} STATIC)

# Add in linked libraries (note necessary syntax for header-only libraries).
target_link_libraries(${PROJECT_NAME} PUBLIC mfem MPI::MPI_CXX)

# Set the source files for the library. 
target_sources(${PROJECT_NAME} PRIVATE src/solvers.cpp 
                                       src/bilininteg.cpp
                                       src/lininteg.cpp                                        
                                       )



option(BUILD_EXAMPLES "Build the example programs" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(GNUInstallDirs)


# sets the search paths for the include files after installation
# as well as during when building the library (as these may differ)
# this allows the library itself and users to #include the library headers
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# this command will append "d" to the name of the debug version of
# the library - this is very helpful when installing as it ensures
# the debug and release version of library can be installed to the
# same location and will not conflict (overwrite each other)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")



# specify the target to install (calculator-static library defined above)
# set the export name <name>-config (does not need to match target name)
# also specify where the .lib file should be installed
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Config
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
    EXPORT ${PROJECT_NAME}Config
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/${PROJECT_NAME}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})


