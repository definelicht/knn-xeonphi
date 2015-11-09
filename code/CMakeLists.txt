cmake_minimum_required(VERSION 2.8)
project(knn)

find_package(Threads REQUIRED)
set(KNN_LIBS ${KNN_LIBS} ${CMAKE_THREAD_LIBS_INIT})

find_package(OpenMP)
if (OpenMP_FOUND OR OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")

set(KNN_SRC
  BinaryIO.cc)

set(KNN_APPS_SRC
  SIFTClassify.cc)

set(KNN_TEST_SRC
  TestBinaryIO.cc
  TestBoundedHeap.cc)

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
endforeach()

add_library(knn ${KNN_SRC_PATHS})
target_link_libraries(knn ${KNN_LIBS})
set(KNN_LIBS ${KNN_LIBS} knn)
foreach(EXEC_PATH ${KNN_EXEC_PATHS})
  get_filename_component(EXEC_FILE_NAME ${EXEC_PATH} NAME_WE)
  add_executable(${EXEC_FILE_NAME} ${EXEC_PATH})
  target_link_libraries(${EXEC_FILE_NAME} knn)
endforeach()
