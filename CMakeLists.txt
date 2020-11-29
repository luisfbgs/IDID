cmake_minimum_required(VERSION 2.8)
project( IDID )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( IDID.out IDID.cpp )
target_link_libraries( IDID.out ${OpenCV_LIBS} )

