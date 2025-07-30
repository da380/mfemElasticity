
find_package(gmsh CONFIG REQUIRED)

add_library(common common.cpp)

add_executable(common_test common_test.cpp )
target_link_libraries(common_test common)

add_executable(offset_disk offset_disk.cpp )
target_link_libraries(offset_disk gmsh common)

add_executable(ball ball.cpp )
target_link_libraries(ball gmsh common)



