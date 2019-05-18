cmake_minimum_required(VERSION 2.8.3)
project(panda_insertion)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  std_srvs
  tf2_msgs
  trajectory_msgs
  roslib
  eigen_conversions
  message_generation
)

find_package(Eigen3 REQUIRED)

add_definitions(-std=c++11) 

add_service_files(FILES
    ChangeState.srv
    SwapController.srv
)

generate_messages()

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs std_srvs tf2_msgs trajectory_msgs roslib eigen_conversions message_runtime
  INCLUDE_DIRS include
  LIBRARIES panda_insertion
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIR})


add_executable(insertion
  src/insertion_main.cpp
  src/Insertion.cpp
  src/Controller.cpp
  src/Panda.cpp
  src/MessageHandler.cpp
)

target_link_libraries(insertion ${catkin_LIBRARIES})

#add_library(panda_insertion
  #src/Insertion.cpp
  #src/Controller.cpp
  #src/Panda.cpp
  #src/MessageHandler.cpp
#)

add_dependencies(insertion ${${PROJECT_NAME}_generate_messages_cpp} ${${PROJECT_NAME}_EXPORT_TARGETS} ${catkin_EXPORTED_TARGETS} )
#add_dependencies(insertion ${PROJECT_NAME}_generate_messages_cpp)

