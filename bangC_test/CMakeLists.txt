cmake_minimum_required(VERSION 2.8)
enable_testing()

################################################################################
# Build Evironment
################################################################################
set(CMAKE_SYSTEM_NAME Linux)
set(BANG_TARGET_CPU_ARCH ${TARGET_CPU_ARCH})
message("-- TARGET_CPU_ARCH=${TARGET_CPU_ARCH}")
set(TARGET_MLU_ARCH ${TARGET_MLU_ARCH})
message("-- TARGET_MLU_ARCH=${TARGET_MLU_ARCH}")
set(TOOLCHAIN_ROOT ${TOOLCHAIN_ROOT})
message("-- TOOLCHAIN_ROOT=${TOOLCHAIN_ROOT}")
set(NEUWARE_HOME ${NEUWARE_HOME})
message("-- NEUWARE_HOME=${NEUWARE_HOME}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${NEUWARE_HOME}/cmake"
  "${NEUWARE_HOME}/cmake/modules"
  )
find_package(BANG)
if(NOT BANG_FOUND)
  message(FATAL_ERROR "BANG cannot be found.")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -fPIC -pthread")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS} -g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${CMAKE_C_FLAGS} -O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -fPIC -std=c++11 -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -g3 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -O3")

set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -Wall -Werror -fPIC -std=c++11 -pthread")
# To be compatible with gcc-4.4.x and libstdc++-4.4.x
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -U __GXX_EXPERIMENTAL_CXX0X__ -U _GLIBCXX_ATOMIC_BUILTINS_4")
if(CMAKE_BUILD_TYPE MATCHES "debug" OR CMAKE_BUILD_TYPE MATCHES "DEBUG")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -g3 -O0")
endif()
if(CMAKE_BUILD_TYPE MATCHES "release" OR CMAKE_BUILD_TYPE MATCHES "RELEASE")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3")
endif()
if(${TARGET_MLU_ARCH} MATCHES "compute_20")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-arch=compute_20")
elseif(${TARGET_MLU_ARCH} MATCHES "compute_30")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-arch=compute_30")
elseif(${TARGET_MLU_ARCH} MATCHES "compute_50")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-arch=compute_50")
else()
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-mlu-arch=${TARGET_MLU_ARCH}")
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/test)

################################################################################
# Build Include
################################################################################
include_directories("${NEUWARE_HOME}/include")

# NOTE: setup cross toolchain as you need
if(NOT ${TARGET_CPU_ARCH} MATCHES "${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
  include_directories("${TOOLCHAIN_ROOT}/aarch64-linux-gnu/include/c++/6.2.1")
  include_directories("${TOOLCHAIN_ROOT}/aarch64-linux-gnu/include/c++/6.2.1/aarch64-linux-gnu")
  link_directories("${TOOLCHAIN_ROOT}/aarch64-linux-gnu/lib64")
endif()

################################################################################
# Build Linking
################################################################################
if(NOT ${TARGET_CPU_ARCH} MATCHES "${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
  link_directories("${NEUWARE_HOME}/edge/lib64")
  link_directories("${NEUWARE_HOME}/edge/lib")
else()
  link_directories("${NEUWARE_HOME}/lib64")
  link_directories("${NEUWARE_HOME}/lib")
endif()

################################################################################
# Function: add unit tests from dir
# Arguments: test_dirname: name of test directory
#            flag_host: flags for host codes
#            flag_bang: flags for bangc codes
#            flag_link: external libraries needed for target executables
################################################################################
function(add_unittests test_dirname flag_host flag_bang flag_link)
  file(GLOB_RECURSE bang_src_files FOLLOW_SYMLINKS ${test_dirname}/*.mlu)
  message(STATUS "-- Enabled unittest: ${test_dirname}")

  foreach(bang_src_file ${bang_src_files})
    get_filename_component(bang_dir_path ${bang_src_file} PATH)
    list(APPEND bang_dir_list ${bang_dir_path})
  endforeach()
  list(REMOVE_DUPLICATES bang_dir_list)

  # parse flag_bang and set BANG_CNCC_FLAGS
  string(REPLACE " " ";" flag_bang_base "${flag_bang}")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} ${flag_bang_base}")

  foreach(test_path ${bang_dir_list})
    get_filename_component(test_name ${test_path} NAME_WE)
    set(binary_name "${test_name}")

    file(GLOB src_bang "${test_path}/*.mlu")
    file(GLOB src_host "${test_path}/*.cpp" "${test_path}/*.cc")

    # host with device binary
    bang_add_executable(${binary_name} ${src_bang} ${src_host})
    set_target_properties(${binary_name} PROPERTIES COMPILE_FLAGS "${flag_host}")
    target_link_libraries(${binary_name} pthread stdc++ m cnrt ${flag_link})

    # binary test
    add_test(NAME ${binary_name} COMMAND ./test/${binary_name})
    set_tests_properties(${binary_name} PROPERTIES PASS_REGULAR_EXPRESSION "PASSED")
    set_tests_properties(${binary_name} PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
  endforeach() # test_path
endfunction()

add_unittests(${CMAKE_SOURCE_DIR}/maxpool
  "-Werror -fPIC -std=c++11 -pthread"
  "-Werror -fPIC -std=c++11"
  "")

add_unittests(${CMAKE_SOURCE_DIR}/add_naive
  "-Werror -fPIC -std=c++11 -pthread"
  "-Werror -fPIC -std=c++11"
  "")

add_unittests(${CMAKE_SOURCE_DIR}/add_3pipeline
  "-Werror -fPIC -std=c++11 -pthread"
  "-Werror -fPIC -std=c++11"
  "")