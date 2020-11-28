cmake_minimum_required(VERSION 2.8)
project( bilinearScale )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( bilinearScale bilinearScale.cpp )
target_link_libraries( bilinearScale ${OpenCV_LIBS} )

