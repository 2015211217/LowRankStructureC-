cmake_minimum_required(VERSION 3.17)
project(LowRankDescendC)

set(CMAKE_CXX_STANDARD 14)

add_executable(LowRankDescendC main.cpp  MVEE.h MVEE.cpp  HedgeAlgorithm.cpp HedgeAlgorithm.h OnlinePCANewVersion.cpp OnlinePCANewVersion.h LowRankExpertAdvice.cpp LowRankExpertAdvice.h Algorithm512.cpp Algorithm512.h SchimidtOrth.cpp SchimidtOrth.h)
#add_executable(LowRankDescendC main.cpp  HedgeAlgorithm.cpp HedgeAlgorithm.h)
#

find_package(OpenCV)
find_package(Eigen3 3.3.9)
find_library(gurobi NAMES gurobi911)

set(OPENCV_ROOT "/usr/local/Cellar/opencv/4.5.1_3")
set(CMAKE_C_FLAGS "-lgmp")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/Cellar/eigen/3.3.9)
include_directories(/Library/gurobi911/mac64/include)
include_directories(/Library/gurobi911/mac64/bin)
include_directories(/Users/jiangxuanke/CLionProjects/LowRankDescendC++/matplotlib-cpp-master)

link_directories(/usr/local/Cellar/opencv/4.5.1_3/lib)
#link_directories(/Library/gurobi911/mac64/lib/libgurobi_c++.a)
link_directories(/Library/gurobi911/mac64/lib)
link_libraries(/Library/gurobi911/mac64/lib/libgurobi_c++.a)
link_libraries(/Library/gurobi911/mac64/lib/libgurobi91.dylib)


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${gurobi911})
target_link_libraries(${PROJECT_NAME} /Library/gurobi911/mac64/lib/libgurobi_c++.a)
target_link_libraries(${PROJECT_NAME} /Library/gurobi911/mac64/lib/libgurobi91.dylib)
target_link_libraries(${PROJECT_NAME} /Library/gurobi911/mac64/lib)




