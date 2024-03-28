import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    params_file = "scenario1.yaml"
    params = os.path.join(
        get_package_share_directory('project1'),
        "config",
        params_file)
    
    ld = launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'ros_g29_force_feedback'), 'launch/g29_feedback.launch.py')
            ),
        ),
        launch_ros.actions.Node(
            package='project1',
            executable='scenario_1',
            output='screen',
            # parameters=[{"use_controller":2}]
        ),
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
