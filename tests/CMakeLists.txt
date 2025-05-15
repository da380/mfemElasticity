include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(TestLinearForm TestLinearForm.cpp)
target_link_libraries(TestLinearForm PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestLinearForm)

add_executable(TestInterpolator TestInterpolator.cpp)
target_link_libraries(TestInterpolator PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestInterpolator)

add_executable(TestBilinearForm TestBilinearForm.cpp)
target_link_libraries(TestBilinearForm PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(TestBilinearForm)


