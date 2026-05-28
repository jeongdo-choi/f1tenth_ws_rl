# F1TENTH Reinforcement Learning Package

A ROS2 package for training and deploying reinforcement learning agents in the F1TENTH simulator.

## Overview

This package provides a framework for implementing reinforcement learning agents for autonomous racing with the F1TENTH simulator. It includes:

- A ROS2 node that interfaces with the F1TENTH simulator
- A reinforcement learning environment wrapper
- DQN (Deep Q-Network) implementation
- Customizable reward functions
- Tools for training and evaluation
- Deployment support for Stable-Baselines3 PPO `.zip` models

## Installation
See setup.md

### Training an RL Agent

1. Start the F1TENTH simulator:
```bash
colcon build # build the workspsace
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

2. In a new terminal, launch the RL agent in training mode:
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_rl rl_agent_launch.py training_mode:=true model_type:=dqn
```

The agent will begin training and save model checkpoints periodically to the specified `save_path` (default: `models/`).

### Deploying a Trained Agent

1. Start the F1TENTH simulator as described above.

2. Launch the RL agent in deployment mode, providing the path to your trained model:
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_rl rl_agent_launch.py training_mode:=false model_path:=/path/to/your/model.pt
```

For a Stable-Baselines3 PPO `.zip` model trained with the `rldd` environment, use `model_type:=sb3_ppo`:
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_rl rl_agent_launch.py \
  training_mode:=false \
  model_type:=sb3_ppo \
  model_path:=/path/to/best_model.zip
```

The SB3 real-car deployment path mirrors the `rldd` training wrappers:
- LaserScan ranges are clipped to `sb3_lidar_max_range`, normalized to `[0, 1]`, and resampled to `sb3_scan_beams` beams.
- The node reads the saved SB3 observation size and action range from the `.zip` metadata when possible.
- 2156-dim models use `[2155 lidar beams, speed / 3.2]`.
- 2160-dim models use `[ang_vels_z, speed / 3.2, linear_vels_s, poses_d, poses_theta, 2155 lidar beams]`.
- For 2160-dim real-car deployment, `sb3_centerline_csv` can point to a centerline or raceline CSV so odometry pose is projected into Frenet-style `poses_d` and `linear_vels_s` features. If no CSV is configured, `poses_d` falls back to `sb3_default_pose_d`.
- The default CSV format is `x,y,yaw`, configured with `sb3_centerline_x_col`, `sb3_centerline_y_col`, and `sb3_centerline_yaw_col`. If your file is `s,x,y,yaw`, set those columns to `1`, `2`, and `3`.
- SB3 actions are mapped from the saved model action range, such as `[-1, -1]..[1, 1]` or `[-1, 0]..[1, 1]`, back to Ackermann steering/speed before publishing `/drive`.
- `drive_max_speed` limits the final command for real-car safety. Keep it low for initial tests.

## Configuration

You can modify the parameters in `config/agent_params.yaml` to adjust:
- Training hyperparameters (learning rate, batch size, etc.)
- Reward function components and weights
- `state_dim`, which is auto-overridden for SB3 checkpoints when the checkpoint input size can be inspected
- SB3 sim-to-real parameters such as `sb3_observation_layout`, `sb3_scan_beams`, `sb3_speed_scale`, `sb3_centerline_csv`, and `drive_max_speed`

## Implementation Details

### RL Agent Node

The `rl_agent_node.py` implements the ROS2 node that:
- Subscribes to laser scan and odometry data from the simulator or vehicle
- Processes observations and calculates rewards
- Trains the reinforcement learning model
- Publishes drive commands to control the vehicle

### Environment Interface

The `environment.py` file provides a wrapper around the simulator that:
- Converts ROS messages into state representations
- Calculates rewards based on driving performance
- Detects episode termination conditions
- Tracks episode progress

### DQN Implementation

The `models/dqn.py` file implements the Deep Q-Network algorithm with:
- Neural network architecture for Q-function approximation
- Experience replay for stable learning
- Target network for reducing overestimation
- Epsilon-greedy exploration strategy

### Reward Function

The `utils/rewards.py` file defines the reward function components:
- Speed rewards for maintaining target velocity
- Progress rewards for lap completion
- Penalties for collisions and excessive steering
- Rewards for centerline following

## Extending the Package

### Adding New RL Algorithms

To implement a new RL algorithm:
1. Create a new file in the `models/` directory
2. Implement the agent class with appropriate methods
3. Update the `rl_agent_node.py` to support the new algorithm
4. Update the parameter file with relevant hyperparameters

### Customizing Reward Functions

You can modify the reward function by:
1. Editing the `calculate_reward` function in `utils/rewards.py`
2. Adjusting reward component weights in the parameter file

## References

- F1TENTH Gym: https://github.com/f1tenth/f1tenth_gym
- F1TENTH Gym ROS Bridge: https://github.com/f1tenth/f1tenth_gym_ros