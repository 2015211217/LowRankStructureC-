cmake_minimum_required(VERSION 3.17)
project(LowRankDescendC__)

set(CMAKE_CXX_STANDARD 14)

add_executable(LowRankDescendC__ MVEE.h MVEE.cpp MirrorDescend.cpp OnlinePCA.h OnlinePCA.cpp HedgeAlgorithm.cpp)