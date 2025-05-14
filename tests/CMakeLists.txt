include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(Testlininteg Testlininteg.cpp)
target_link_libraries(Testlininteg PRIVATE  mfemElasticity gtest_main)
gtest_discover_tests(Testlininteg)
