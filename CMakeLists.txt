cmake_minimum_required(VERSION 3.5)
project(lidar_cam_fusion)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install the Python package so you can do "from lidar_cam_fusion import …"
ament_python_install_package(${PROJECT_NAME})

# Install launch and config resources
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Install the executable script into lib/<pkg> so launch can find it
install(PROGRAMS
  lidar_cam_fusion/fusion_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# (Optional) resource marker / package.xml
install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
