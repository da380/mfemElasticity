#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "mfemElasticity::mfemElasticity" for configuration ""
set_property(TARGET mfemElasticity::mfemElasticity APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(mfemElasticity::mfemElasticity PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libmfemElasticity.a"
  )

list(APPEND _cmake_import_check_targets mfemElasticity::mfemElasticity )
list(APPEND _cmake_import_check_files_for_mfemElasticity::mfemElasticity "${_IMPORT_PREFIX}/lib/libmfemElasticity.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
