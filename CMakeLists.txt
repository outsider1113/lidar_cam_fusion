cmake_minimum_required(VERSION 3.5)
project(lidar_cam_fusion)

# Build tools
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Interface generation and message dependencies
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Declare and generate message interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FusionDetections.msg"
  "msg/DetectedObject.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

# Install the Python package so “import lidar_cam_fusion” works
ament_python_install_package(${PROJECT_NAME})

# Install launch and config resources
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Install the executable node script (fusion_node.py)
install(PROGRAMS
  lidar_cam_fusion/fusion_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# (Optional) install package.xml into share
install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

# Export runtime dependency on generated messages
ament_export_dependencies(rosidl_default_runtime)

ament_package()
