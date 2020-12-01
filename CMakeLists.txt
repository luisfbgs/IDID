cmake_minimum_required(VERSION 2.8)
project( IDID )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} include )
file(GLOB SOURCES "src/*.cpp")
add_executable( IDID.out ${SOURCES} )
target_link_libraries( IDID.out ${OpenCV_LIBS} )

