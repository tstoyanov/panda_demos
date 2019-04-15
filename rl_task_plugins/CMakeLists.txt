cmake_minimum_required(VERSION 2.8.3)
project(rl_task_plugins)

set(CMAKE_BUILD_TYPE RelWithDebinfo)
set(CMAKE_CXX_STANDARD 11)

set(GUROBI_INCLUDE_DIR "$ENV{GUROBI_HOME}/include")
set(GUROBI_LIB_DIR "$ENV{GUROBI_HOME}/lib")
set(GUROBI_LIBS gurobi_c++ gurobi75)

add_definitions(-Wno-missing-braces)

find_package(catkin REQUIRED COMPONENTS
  hiqp_core
  hiqp_ros
  pluginlib
  roscpp
  message_generation
)

add_service_files(
        FILES
        Act.srv
        )

generate_messages()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rl_task_plugins
  CATKIN_DEPENDS hiqp_core hiqp_ros pluginlib roscpp message_runtime
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GUROBI_INCLUDE_DIR}
)

link_directories(${GUROBI_LIB_DIR})

add_library(${PROJECT_NAME}_tdef src/tdef_rl_full_pose.cpp)
add_dependencies(${PROJECT_NAME}_tdef ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_tdef ${catkin_LIBRARIES} ${GUROBI_LIBS})


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

