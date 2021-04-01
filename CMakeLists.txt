cmake_minimum_required(VERSION 3.17)
project(LowRankDescendC)

set(CMAKE_CXX_STANDARD 14)

add_executable(LowRankDescendC main.cpp MirrorDescend.h MirrorDescend.cpp HedgeAlgorithm.cpp HedgeAlgorithm.h MVEE.cpp MVEE.h OnlinePCA.cpp OnlinePCA.h)

find_package(OpenCV)
find_package(Eigen3 3.3.9)

set(OPENCV_ROOT "/usr/local/Cellar/opencv/4.5.1_3")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/Cellar/eigen/3.3.9)

link_directories(/usr/local/Cellar/opencv/4.5.1_3/lib)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)