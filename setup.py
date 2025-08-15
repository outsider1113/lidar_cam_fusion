from setuptools import setup

package_name = 'lidar_cam_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/fusion.launch.py']),
        ('share/' + package_name + '/config', ['config/fusion_config.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Angel Hernandez',
    maintainer_email='angeljoel064@gmail.com',
    description='LiDAR × Camera fusion node Foxy',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fusion_node = lidar_cam_fusion.fusion_node:main',
        ],
    },
)
