cmake_minimum_required(VERSION 2.8.3)
project(hand)

find_package(catkin REQUIRED)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
  cv_bridge
  image_transport
  pcl_ros
  pcl_conversions
  sensor_msgs
)


catkin_package(
)


add_executable(hand src/main.cpp)
target_link_libraries(hand ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})