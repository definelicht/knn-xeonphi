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
  set(Vc_ALL_FLAGS_STRING "${Vc_ALL_FLAGS}")
  string(REPLACE ";" " " Vc_ALL_FLAGS_STRING "${Vc_ALL_FLAGS_STRING}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Vc_ALL_FLAGS_STRING} -DKNN_USE_VC")
endif()

find_package(TBB REQUIRED)
if (NOT TBB_MALLOC_FOUND)
  message(FATAL_ERROR "TBB malloc library required for scalable allocator.")
endif()
include_directories(AFTER SYSTEM ${TBB_INCLUDE_DIRS} ${TBB_MALLOC_INCLUDE_DIRS})
set(KNN_LIBS ${KNN_LIBS} ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARIES})
if (TBB_MALLOC_PROXY_FOUND)
  include_directories(AFTER SYSTEM ${TBB_MALLOC_PROXY_INCLUDE_DIRS})
  set(KNN_LIBS ${KNN_LIBS} ${TBB_MALLOC_PROXY_LIBRARIES})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Weffc++")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
endif()

set(KNN_SRC
  BinaryIO.cc)

set(KNN_APPS_SRC
  SIFTClassify.cc
  TestTreeBuildParallel.cpp)

set(KNN_TEST_SRC
  TestBinaryIO.cc
  TestBoundedHeap.cc
  TestTraversal.cc
  TestVariance.cc)

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
