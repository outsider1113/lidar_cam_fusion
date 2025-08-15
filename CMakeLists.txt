cmake_minimum_required(VERSION 3.5)
project(lidar_cam_fusion)

# Using Python packaging via ament_python
find_package(ament_cmake REQUIRED)
find_package(ament_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install launch and config resources
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# (Optional) resource marker so ament_index can find the share dir
install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
