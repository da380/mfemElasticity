
find_package(gmsh CONFIG REQUIRED)

add_library(meshing CircularMesh.cpp)
target_link_libraries(meshing PUBLIC gmsh)

add_executable(meshing_test meshing_test.cpp )
target_link_libraries(meshing_test meshing)

add_library(common common.cpp)

add_executable(offset_disk offset_disk.cpp )
target_link_libraries(offset_disk gmsh common)

add_executable(ball ball.cpp )
target_link_libraries(ball gmsh common)



