cmake_minimum_required(VERSION 2.8)
project(knn)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules/)

option(KNN_USE_VC "Use Vc library for explicit vectorization if available." ON)
option(KNN_USE_OPENMP "Enable OpenMP if available." ON)
option(KNN_USE_FLANN "Compile with FLANN if available." ON)
option(KNN_USE_MPI "Compile with MPI if available." ON)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  if (CMAKE_BUILD_TYPE AND (NOT CMAKE_BUILD_TYPE STREQUAL "Release"))
    message(WARNING "Unknown build type. Defaulting to release.")
  endif()
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(Threads REQUIRED)
set(KNN_LIBS ${KNN_LIBS} ${CMAKE_THREAD_LIBS_INIT})

if (KNN_USE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_FOUND OR OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKNN_USE_OMP ${OpenMP_CXX_FLAGS}")
  endif()
endif()

if (KNN_USE_VC)
  find_package(Vc)
  if (Vc_FOUND)
    include_directories(AFTER SYSTEM ${Vc_INCLUDE_DIR})
    set(KNN_LIBS ${KNN_LIBS} ${Vc_LIBRARIES})
    set(Vc_ALL_FLAGS_STRING "${Vc_ALL_FLAGS}")
    string(REPLACE ";" " " Vc_ALL_FLAGS_STRING "${Vc_ALL_FLAGS_STRING}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Vc_ALL_FLAGS_STRING} -DKNN_USE_VC")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
endif()

if (KNN_USE_MPI)
  find_package(MPI)
  if (MPI_CXX_FOUND)
    message(STATUS "Building WITH MPI support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKNN_USE_MPI")
    set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    set(KNN_LIBS ${KNN_LIBS} ${MPI_CXX_LIBRARIES})
    set(KNN_APPS_SRC ${KNN_APPS_SRC} SIFTClassifyMPI.cc)
  else()
    message(STATUS "Building WITHOUT MPI support.")
  endif()
endif()

if (NOT TBB_LIBRARY_PATH AND NOT TBB_LIBRARIES AND NOT TBB_INCLUDE_PATH)
  find_package(TBB REQUIRED)
  if (NOT TBB_MALLOC_FOUND)
    message(FATAL_ERROR "TBB malloc library required for scalable allocator.")
  endif()
  include_directories(AFTER SYSTEM ${TBB_INCLUDE_DIRS} ${TBB_MALLOC_INCLUDE_DIRS})
  set(KNN_LIBS ${KNN_LIBS} ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARIES})
  message(STATUS "TBB libraries found: ${TBB_LIBRARIES}")
  if (TBB_MALLOC_PROXY_FOUND)
    include_directories(AFTER SYSTEM ${TBB_MALLOC_PROXY_INCLUDE_DIRS})
    set(KNN_LIBS ${KNN_LIBS} ${TBB_MALLOC_PROXY_LIBRARIES})
  endif()
else()
  message(STATUS "Using user-provided TBB include path: ${TBB_INCLUDE_PATH}")
  message(STATUS "Using user-provided TBB library path: ${TBB_LIBRARY_PATH}")
  message(STATUS "Using user-provided TBB libraries: ${TBB_LIBRARIES}")
  include_directories(AFTER SYSTEM ${TBB_INCLUDE_PATH})
  link_directories(${TBB_LIBRARY_PATH})
  set(KNN_LIBS ${KNN_LIBS} ${TBB_LIBRARIES})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
if (MIC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic -DKNN_USE_MIC")
endif()
if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
endif()

if (KNN_USE_FLANN)
  set(KNN_SRC
    flann/ext/lz4.c
    flann/ext/lz4hc.c)
  set(KNN_APPS_SRC ${KNN_APPS_SRC}
    TestFlann.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKNN_USE_FLANN")
endif()

message(STATUS "Compiling with flags: ${CMAKE_CXX_FLAGS}")

set(KNN_SRC ${KNN_SRC}
    ParseArguments.cpp)

set(KNN_APPS_SRC ${KNN_APPS_SRC}
    SIFTClassify.cc
    TestTreeBuildParallel.cpp)

set(KNN_TEST_SRC ${KNN_TEST_SRC}
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

if (KNN_SRC_PATHS)
  add_library(knn ${KNN_SRC_PATHS})
  target_link_libraries(knn ${KNN_LIBS})
  set(KNN_LIBS ${KNN_LIBS} knn)
endif()
foreach(EXEC_PATH ${KNN_EXEC_PATHS})
  get_filename_component(EXEC_FILE_NAME ${EXEC_PATH} NAME_WE)
  if (MIC)
    add_executable("${EXEC_FILE_NAME}.mic" ${EXEC_PATH})
    target_link_libraries("${EXEC_FILE_NAME}.mic" ${KNN_LIBS})
  else()
    add_executable(${EXEC_FILE_NAME} ${EXEC_PATH})
    target_link_libraries(${EXEC_FILE_NAME} ${KNN_LIBS})
  endif()
endforeach()
