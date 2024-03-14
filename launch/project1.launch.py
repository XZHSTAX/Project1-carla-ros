import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
          
    ld = launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'ros_g29_force_feedback'), 'g29_feedback.launch.py')
            ),
        ),
        launch_ros.actions.Node(
            package='project1',
            executable='scenario_1',
            output='screen'
        ),
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
