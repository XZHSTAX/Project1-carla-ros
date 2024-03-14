from setuptools import setup
import os
from glob import glob
package_name = 'project1'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,'project1.util'],
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xzh',
    maintainer_email='XZHSTAX@foxmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 'scenario_1 = project1.scenario_1:main'
        ],
    },
)
