cmake_minimum_required(VERSION 3.2)
project(DSLDA)

add_subdirectory(gflags)

set (CMAKE_CXX_FLAGS "-march=native -O3 -std=c++1y -Wall -g ${CMAKE_CXX_FLAGS}")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

include(FindOpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif (OPENMP_FOUND)

add_subdirectory(lda)
