cmake_minimum_required(VERSION 2.8)
project(knn)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules/)

find_package(Threads REQUIRED)
set(KNN_LIBS ${KNN_LIBS} ${CMAKE_THREAD_LIBS_INIT})

find_package(OpenMP)
if (OpenMP_FOUND OR OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKNN_USE_OMP ${OpenMP_CXX_FLAGS}")
endif()

find_package(Vc)
if (Vc_FOUND)
  include_directories(AFTER SYSTEM ${Vc_INCLUDE_DIR})
  set(KNN_LIBS ${KNN_LIBS} ${Vc_LIBRARIES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKNN_USE_VC")
endif()

find_package(TBB)
if (TBB_FOUND)
  include_directories(AFTER SYSTEM ${TBB_INCLUDE_DIRS})
  set(KNN_LIBS ${KNN_LIBS} ${TBB_LIBRARIES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_TBB_")
endif()
if (TBB_MALLOC_PROXY_FOUND)
  include_directories(AFTER SYSTEM ${TBB_MALLOC_PROXY_INCLUDE_DIRS})
  set(KNN_LIBS ${KNN_LIBS} ${TBB_MALLOC_PROXY_LIBRARIES})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")

set(KNN_SRC
  BinaryIO.cc)

set(KNN_APPS_SRC
  SIFTClassify.cc)

set(KNN_TEST_SRC
  TestBinaryIO.cc
  TestBoundedHeap.cc
  TestTraversal.cc
  TestVariance.cc)

if (OpenMP_FOUND)
  set(KNN_TEST_SRC ${KNN_TEST_SRC} TestTreeBuildParallel.cpp)
endif()

# Enable CTest for testing
enable_testing()

set(KNN_SRC_PATHS)
foreach(SRC_FILE ${KNN_SRC})
  set(KNN_SRC_PATHS ${KNN_SRC_PATHS}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/${SRC_FILE})
endforeach()
foreach(SRC_FILE ${KNN_APPS_SRC})
  set(KNN_EXEC_PATHS ${KNN_EXEC_PATHS}
      ${CMAKE_CURRENT_SOURCE_DIR}/apps/${SRC_FILE})
endforeach()
foreach(SRC_FILE ${KNN_TEST_SRC})
  set(KNN_EXEC_PATHS ${KNN_EXEC_PATHS}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/${SRC_FILE})
  get_filename_component(TEST_FILE_NAME ${SRC_FILE} NAME_WE)
  add_test(${TEST_FILE_NAME} ${TEST_FILE_NAME})
endforeach()

add_library(knn ${KNN_SRC_PATHS})
target_link_libraries(knn ${KNN_LIBS})
set(KNN_LIBS ${KNN_LIBS} knn)
foreach(EXEC_PATH ${KNN_EXEC_PATHS})
  get_filename_component(EXEC_FILE_NAME ${EXEC_PATH} NAME_WE)
  add_executable(${EXEC_FILE_NAME} ${EXEC_PATH})
  target_link_libraries(${EXEC_FILE_NAME} ${KNN_LIBS})
endforeach()
