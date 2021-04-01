cmake_minimum_required(VERSION 3.17)
project(LowRankDescendC)

set(CMAKE_CXX_STANDARD 14)


add_executable(LowRankDescendC main.cpp MirrorDescend.h MirrorDescend.cpp HedgeAlgorithm.cpp HedgeAlgorithm.h MVEE.cpp MVEE.h OnlinePCA.cpp OnlinePCA.h)

find_package(OpenCV)
set(OPENCV_ROOT "/usr/local/Cellar/opencv")
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(/usr/local/Cellar/opencv/4.5.1_3/lib)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})