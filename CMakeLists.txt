include_directories(.)
cmake_minimum_required(VERSION 3.16)
project(main)
find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    )
    link_directories(
        ${OpenCV_LIBS}
        )
include_directories(.)
add_executable(main src/main.cpp src/OpenCV.cpp src/test.cpp)
target_link_libraries(main  ${OpenCV_LIBS})

