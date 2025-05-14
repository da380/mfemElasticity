

add_executable(ex1 ex1.cpp)
target_link_libraries(ex1 mfemElasticity)

if(USE_MPI)
        add_executable(ex1p ex1p.cpp)
        target_link_libraries(ex1p mfemElasticity)
endif()

add_executable(ex2 ex2.cpp)
target_link_libraries(ex2 mfemElasticity)


