cmake_minimum_required(VERSION 2.8.3)
project(panda_insertion)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  trajectory_msgs
)

## Declare a catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES panda_insertion
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  include
)

add_definitions(-std=c++11) 

add_library(panda_insertion
  src/StateMachine.cpp
  src/Controller.cpp
)

add_executable(insertion
  src/insertion.cpp
  src/StateMachine.cpp
  src/Controller.cpp
)
target_link_libraries(insertion ${catkin_LIBRARIES})
add_dependencies(insertion panda_insertion_cpp)