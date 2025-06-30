

add_executable(ex1 ex1.cpp)
target_link_libraries(ex1 mfemElasticity)

if(USE_MPI)
        add_executable(ex1p ex1p.cpp)
        target_link_libraries(ex1p mfemElasticity)
endif()

add_executable(ex2 ex2.cpp)
target_link_libraries(ex2 mfemElasticity)


add_executable(ex3 ex3.cpp)
target_link_libraries(ex3 mfemElasticity)

add_executable(ex4 ex4.cpp)
target_link_libraries(ex4 mfemElasticity)



