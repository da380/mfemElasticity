
find_package(gmsh CONFIG REQUIRED)

add_executable(disk disk.cpp)
target_link_libraries(disk gmsh)

add_executable(ball ball.cpp)
target_link_libraries(ball gmsh)



