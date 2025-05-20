cmake_minimum_required(VERSION 3.15)

project(
    mfemElasticity-apss
    VERSION 0.0.1
    LANGUAGES CXX)


find_package(mfemElasticity REQUIRED CONFIG)



add_executable(app1)
target_sources(app1 PRIVATE app1.cpp)

target_link_libraries(app1
                      PUBLIC  mfemElasticity)

