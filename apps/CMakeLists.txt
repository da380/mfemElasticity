cmake_minimum_required(VERSION 3.15)

project(
    mfemElasticity-apps
    VERSION 0.0.1
    LANGUAGES CXX)


find_package(mfemElasticity REQUIRED CONFIG)
message(STATUS "Found mfemElasticity config in: ${mfemElasticity_DIR} (version ${mfemElasticity_VERSION})")


add_executable(app1)
target_sources(app1 PRIVATE app1.cpp)


target_link_libraries(app1
                      PUBLIC mfemElasticity)

if(USE_MPI)                    
    add_executable(app1p)
    target_sources(app1p PRIVATE app1p.cpp)
    target_link_libraries(app1p
                      PUBLIC  mfemElasticity)
endif()


