cmake_minimum_required(VERSION 3.0)
project(myTest)

find_package(OpenMP)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(/usr/include/malloc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wall -Wextra -Weffc++ ${OpenMP_CXX_FLAGS}")

set(KNN_SRC
    jpge.cpp)

set(KNN_SRC_PATHS)
foreach(SRC_FILE ${KNN_SRC})
  set(KNN_SRC_PATHS ${KNN_SRC_PATHS} ${CMAKE_CURRENT_SOURCE_DIR}/src/${SRC_FILE})
endforeach()

add_library(myLib ${KNN_SRC_PATHS})
target_link_libraries(myLib ${KNN_LIBS})
set(KNN_LIBS ${KNN_LIBS} myLib)
add_executable(myKNNTest ${CMAKE_CURRENT_SOURCE_DIR}/Test_KNN.cpp)
target_link_libraries(myKNNTest myLib)
add_executable(myKDTreeTest ${CMAKE_CURRENT_SOURCE_DIR}/Test_KDTree.cpp)
target_link_libraries(myKDTreeTest myLib)
