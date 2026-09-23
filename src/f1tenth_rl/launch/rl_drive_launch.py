"""Run the packaged SB3 PPO policy; start vehicle bringup separately."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('f1tenth_rl')
    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(package_share, 'config', 'agent_params.yaml'),
        description='Vehicle-specific PPO preprocessing and drive parameters',
    )
    model_path = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(package_share, 'models', 'best_model.zip'),
        description='Stable-Baselines3 PPO checkpoint, normally the packaged model',
    )
    agent = Node(
        package='f1tenth_rl',
        executable='rl_agent',
        name='rl_agent',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'training_mode': False,
                'model_type': 'sb3_ppo',
                'model_path': LaunchConfiguration('model_path'),
                'save_path': os.path.expanduser('~/.ros/f1tenth_rl'),
            },
        ],
        output='screen',
    )
    return LaunchDescription([params_file, model_path, agent])
