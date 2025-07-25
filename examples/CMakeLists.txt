

add_executable(ex1 ex1.cpp)
target_link_libraries(ex1 mfemElasticity)

add_executable(ex2 ex2.cpp)
target_link_libraries(ex2 mfemElasticity)

add_executable(ex3 ex3.cpp)
target_link_libraries(ex3 mfemElasticity)

add_executable(ex4 ex4.cpp)
target_link_libraries(ex4 mfemElasticity)

add_executable(ex5 ex5.cpp)
target_link_libraries(ex5 mfemElasticity)

if(USE_MPI)
        add_executable(ex1p ex1p.cpp)
        target_link_libraries(ex1p mfemElasticity)

        add_executable(ex2p ex2p.cpp)
        target_link_libraries(ex2p mfemElasticity)

        add_executable(ex3p ex3p.cpp)
        target_link_libraries(ex3p mfemElasticity)

        add_executable(ex4p ex4p.cpp)
        target_link_libraries(ex4p mfemElasticity)

        add_executable(ex5p ex5p.cpp)
        target_link_libraries(ex5p mfemElasticity)
endif()
