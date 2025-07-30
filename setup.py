from setuptools import setup

package_name = 'lidar_cam_fusion'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/lidar_cam_fusion']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/fusion_config.yaml', 'config/sim_box.pt']),
        ('share/' + package_name + '/launch', ['launch/fusion_launch.py']),
        ('lib/' + package_name, ['lidar_cam_fusion/fusion_node.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Angel Hernandez',
    maintainer_email='angeljoel064@gmail.com',
    description='LiDAR and Camera Fusion Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],  # Disabled
    },
)