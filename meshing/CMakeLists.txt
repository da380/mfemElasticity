
find_package(gmsh CONFIG REQUIRED)



add_library(meshing CircularMesh.cpp SphereMesh.cpp common.cpp)
target_link_libraries(meshing PUBLIC gmsh::shared)

add_executable(circle_test circle_test.cpp )
target_link_libraries(circle_test meshing)

add_executable(sphere_test sphere_test.cpp )
target_link_libraries(sphere_test meshing)

add_executable(disk disk.cpp )
target_link_libraries(disk  meshing)

add_executable(disk2 disk2.cpp )
target_link_libraries(disk2  meshing)

add_executable(diskn diskn.cpp )
target_link_libraries(diskn  meshing)


add_executable(offset_disk offset_disk.cpp )
target_link_libraries(offset_disk  meshing)

add_executable(ball ball.cpp )
target_link_libraries(ball meshing)

add_executable(offset_ball offset_ball.cpp )
target_link_libraries(offset_ball meshing)

add_executable(geomesh_prem_crust geomesh_prem_crust.cpp )
target_link_libraries(geomesh_prem_crust meshing mfemElasticity)



