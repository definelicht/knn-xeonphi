cmake_minimum_required(VERSION 3.0)
project(Knn)

find_package(OpenMP)
include_directories({$CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -Weffc++ ${OpenMP_CXX_FLAGS}")
add_executable(TestKnn ${CMAKE_CURRENT_SOURCE_DIR}/TestKnn.cc)
add_executable(TestKDTree ${CMAKE_CURRENT_SOURCE_DIR}/TestKDTree.cc)
