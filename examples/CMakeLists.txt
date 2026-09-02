
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

add_executable(ex6 ex6.cpp)
target_link_libraries(ex6 mfemElasticity)

add_executable(ex7 ex7.cpp)
target_link_libraries(ex7 mfemElasticity)

add_executable(ex8 ex8.cpp)
target_link_libraries(ex8 mfemElasticity)

add_executable(ex9 ex9.cpp)
target_link_libraries(ex9 mfemElasticity)

add_executable(ex10 ex10.cpp)
target_link_libraries(ex10 mfemElasticity)

add_executable(grid_transfer grid_transfer.cpp)
target_link_libraries(grid_transfer mfemElasticity)

add_executable(ex11 ex11.cpp)
target_link_libraries(ex11 mfemElasticity)

add_executable(ex12 ex12.cpp)
target_link_libraries(ex12 mfemElasticity)


add_executable(quasi_static_elasticity quasi_static_elasticity.cpp)
target_link_libraries(quasi_static_elasticity mfemElasticity)

add_executable(viscoelasticity viscoelasticity.cpp)
target_link_libraries(viscoelasticity mfemElasticity)


add_executable(wave wave.cpp)
target_link_libraries(wave mfemElasticity)

add_executable(ode ode.cpp)
target_link_libraries(ode mfemElasticity)

add_executable(seismic seismic.cpp)
target_link_libraries(seismic mfemElasticity)

add_executable(submesh_injection submesh_injection.cpp)
target_link_libraries(submesh_injection mfemElasticity)



if(USE_MPI)
        add_executable(ex1p ex1p.cpp)
        target_link_libraries(ex1p mfemElasticity)

        add_executable(ex2p ex2p.cpp)
        target_link_libraries(ex2p mfemElasticity)

        add_executable(ex3p ex3p.cpp)
        target_link_libraries(ex3p mfemElasticity)

        add_executable(ex4p ex4p.cpp)
        target_link_libraries(ex4p mfemElasticity)

        add_executable(ex7p ex7p.cpp)
        target_link_libraries(ex7p mfemElasticity)

        add_executable(ex8p ex8p.cpp)
        target_link_libraries(ex8p mfemElasticity)

        add_executable(ex9p ex9p.cpp)
        target_link_libraries(ex9p mfemElasticity)

        add_executable(ex10p ex10p.cpp)
        target_link_libraries(ex10p mfemElasticity)

        add_executable(grid_transfer_p grid_transfer_p.cpp)
        target_link_libraries(grid_transfer_p mfemElasticity)

        add_executable(seismic_p seismic_p.cpp)
        target_link_libraries(seismic_p mfemElasticity)

        add_executable(ex11p ex11p.cpp)
        target_link_libraries(ex11p mfemElasticity)

        add_executable(ex12p ex12p.cpp)
        target_link_libraries(ex12p mfemElasticity)

        add_executable(submesh_injection_p submesh_injection_p.cpp)
        target_link_libraries(submesh_injection_p mfemElasticity)
endif()
