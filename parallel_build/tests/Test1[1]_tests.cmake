add_test([=[Test.One]=]  /home/david/dev/mfemElasticity/parallel_build/tests/Test1 [==[--gtest_filter=Test.One]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Test.One]=]  PROPERTIES WORKING_DIRECTORY /home/david/dev/mfemElasticity/parallel_build/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  Test1_TESTS Test.One)
