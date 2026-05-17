#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import torch
import os
from datetime import datetime
from f1tenth_rl.environment import F1TenthEnv
from f1tenth_rl.models.dqn import DQNAgent
from f1tenth_rl.models.ppo import PPOAgent
from f1tenth_rl.utils.rewards import calculate_reward

class RLAgentNode(Node):
    def __init__(self):
        super().__init__('rl_agent_node')
        
        # Declare parameters
        self.declare_parameter('training_mode', True)
        self.declare_parameter('model_type', 'dqn')
        self.declare_parameter('model_path', '')
        self.declare_parameter('save_path', 'models/')
        self.declare_parameter('state_dim', 1080)
        self.declare_parameter('max_steps_per_episode', 1000)
        self.declare_parameter('hidden_dim', 256)
        self.declare_parameter('learning_rate', 0.0001)
        self.declare_parameter('batch_size', 64)
        self.declare_parameter('gamma', 0.98)
        self.declare_parameter('epsilon_start', 1.0)
        self.declare_parameter('epsilon_end', 0.05)
        self.declare_parameter('epsilon_decay', 0.99)
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_yaw', 0.0)
        
        # Get parameters
        self.training_mode = self.get_parameter('training_mode').value
        self.model_type = self.get_parameter('model_type').value.lower().replace('-', '_')
        self.model_path = self.get_parameter('model_path').value
        self.save_path = self.get_parameter('save_path').value
        self.state_dim = int(self.get_parameter('state_dim').value)
        self.max_steps_per_episode = int(self.get_parameter('max_steps_per_episode').value)
        self.hidden_dim = int(self.get_parameter('hidden_dim').value)
        self.learning_rate = float(self.get_parameter('learning_rate').value)
        self.batch_size = int(self.get_parameter('batch_size').value)
        self.gamma = float(self.get_parameter('gamma').value)
        self.epsilon_start = float(self.get_parameter('epsilon_start').value)
        self.epsilon_end = float(self.get_parameter('epsilon_end').value)
        self.epsilon_decay = float(self.get_parameter('epsilon_decay').value)
        
        # Create directories if they don't exist
        os.makedirs(self.save_path, exist_ok=True)
        
        # Publishers and subscribers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/pf/pose/odom', self.odom_callback, 10)
        
        # Store the latest observations
        self.latest_scan = None
        self.latest_odom = None
        self.prev_odom = None
        
        # Initialize environment with reference to this node
        self.env = F1TenthEnv(node=self)
        
        # Set starting position from parameters
        self.env.start_position = [
            self.get_parameter('start_x').value,
            self.get_parameter('start_y').value,
            self.get_parameter('start_yaw').value
        ]
        
        # Initialize RL agent
        self.initialize_agent()
        
        # Training loop timer (runs at 10Hz)
        self.timer = self.create_timer(0.1, self.training_loop)
        
        # Episodic data
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episodes_completed = 0
        self.pending_transition = None
        self.ppo_update_interval = 200
        
        self.get_logger().info('RL Agent Node initialized')
    
    def initialize_agent(self):
        """Initialize the RL agent based on specified model type"""
        if self.model_type == 'dqn':
            self.agent = DQNAgent(
                state_dim=self.state_dim,
                action_dim=15,
                hidden_dim=self.hidden_dim,
                learning_rate=self.learning_rate
            )
            self.agent.batch_size = self.batch_size
            self.agent.gamma = self.gamma
            self.agent.epsilon = self.epsilon_start
            self.agent.epsilon_min = self.epsilon_end
            self.agent.epsilon_decay = self.epsilon_decay
            self.actions = self._build_discrete_actions()
            self.get_logger().info(f'Initialized DQN agent with {len(self.actions)} discrete actions')
        elif self.model_type == 'ppo':
            self.agent = PPOAgent(
                state_dim=self.state_dim,
                action_dim=2,
                hidden_dim=self.hidden_dim,
                lr=self.learning_rate
            )
            self.actions = None
            self.get_logger().info('Initialized custom PPO agent with continuous actions')
        elif self.model_type in ('sb3_ppo', 'stable_baselines3_ppo'):
            if self.training_mode:
                raise ValueError('SB3 PPO mode is for deployment only. Launch with training_mode:=false.')
            if not self.model_path:
                raise ValueError('SB3 PPO mode requires model_path to point to a Stable-Baselines3 .zip model.')

            from stable_baselines3 import PPO as SB3PPO
            self.agent = SB3PPO.load(self.model_path)
            self.actions = None
            self.get_logger().info(f'Loaded Stable-Baselines3 PPO model from {self.model_path}')
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        # Load custom PyTorch checkpoints if provided.
        if self.model_path and self.model_type not in ('sb3_ppo', 'stable_baselines3_ppo'):
            try:
                self.agent.load(self.model_path)
                self.get_logger().info(f'Loaded model from {self.model_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to load model: {e}')
    
    def _build_discrete_actions(self):
        """Create the steering/velocity grid used by DQN."""
        actions = []
        for steering in np.linspace(-0.4, 0.4, 5):
            for velocity in [1.0, 2.0, 3.0]:
                actions.append([steering, velocity])
        return actions
    
    def scan_callback(self, msg):
        """Store the latest laser scan data"""
        self.latest_scan = msg
    
    def odom_callback(self, msg):
        """Store the latest odometry data"""
        self.prev_odom = self.latest_odom
        self.latest_odom = msg
    
    def get_state(self):
        """Convert laser scan to a fixed-size state vector for the RL agent"""
        if self.latest_scan is None:
            return None
            
        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=10.0, posinf=10.0, neginf=0.0)
        normalized_ranges = np.clip(ranges / 10.0, 0.0, 1.0)

        if len(normalized_ranges) > self.state_dim:
            normalized_ranges = normalized_ranges[:self.state_dim]
        elif len(normalized_ranges) < self.state_dim:
            normalized_ranges = np.pad(
                normalized_ranges,
                (0, self.state_dim - len(normalized_ranges)),
                mode='constant',
                constant_values=1.0
            )
        
        return normalized_ranges
    
    def publish_drive_command(self, steering, velocity):
        """Publish drive command to the car"""
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(np.clip(steering, -0.4, 0.4))
        msg.drive.speed = float(np.clip(velocity, 0.0, 2.0))
        self.drive_pub.publish(msg)

    def select_action(self, state):
        """Select an action and return metadata needed for training."""
        if self.model_type == 'dqn':
            if self.training_mode and np.random.random() < self.agent.epsilon:
                action_idx = np.random.randint(0, len(self.actions))
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_idx = self.agent.select_action(state_tensor)

            steering, velocity = self.actions[action_idx]
            return steering, velocity, {'action_idx': action_idx}

        if self.model_type in ('sb3_ppo', 'stable_baselines3_ppo'):
            action, _ = self.agent.predict(state, deterministic=True)
            action = np.asarray(action).reshape(-1)
            if action.size < 2:
                raise ValueError(f'SB3 PPO action must contain steering and velocity, got shape {action.shape}')
            return action[0], action[1], {}

        deterministic = not self.training_mode
        action, log_prob, value = self.agent.select_action(state, deterministic=deterministic)
        steering, velocity = action
        return steering, velocity, {
            'action': action,
            'log_prob': log_prob,
            'value': value,
        }
    
    def training_loop(self):
        """Main RL training/inference loop"""
        if self.latest_scan is None:
            return
        if self.training_mode and self.latest_odom is None:
            return
        
        current_state = self.get_state()
        if current_state is None:
            return

        if self.training_mode:
            self._learn_from_pending_transition(current_state)

        if self.pending_transition is not None:
            return

        steering, velocity, metadata = self.select_action(current_state)
        self.publish_drive_command(steering, velocity)

        if self.training_mode:
            self.pending_transition = {
                'state': current_state.copy(),
                **metadata,
            }

    def _learn_from_pending_transition(self, next_state):
        """Store and train on the action issued during the previous timer tick."""
        if self.pending_transition is None or self.prev_odom is None:
            return

        reward, done = calculate_reward(
            self.latest_scan,
            self.latest_odom,
            self.prev_odom
        )
        
        self.episode_reward += reward
        self.episode_steps += 1

        if self.model_type == 'dqn':
            self.agent.store_transition(
                self.pending_transition['state'],
                self.pending_transition['action_idx'],
                reward,
                next_state.copy(),
                done
            )
            
            if len(self.agent.replay_buffer) > self.agent.batch_size:
                self.agent.train()
        elif self.model_type == 'ppo':
            self.agent.store_transition(
                self.pending_transition['state'],
                self.pending_transition['action'],
                self.pending_transition['log_prob'],
                reward,
                self.pending_transition['value'],
                done
            )

            if self.episode_steps % self.ppo_update_interval == 0 or done:
                next_value = 0.0
                if not done:
                    _, _, next_value = self.agent.select_action(next_state, deterministic=True)
                self.agent.train(next_value)

        self.pending_transition = None

        if done or self.episode_steps >= self.max_steps_per_episode:
            self._finish_episode()

    def _finish_episode(self):
        """Log, save, and reset after an episode ends."""
        self.episodes_completed += 1
        self.env.reset_car_position()
        self.get_logger().info('Episode ended! Resetting car position.')
        
        self.get_logger().info(
            f'Episode {self.episodes_completed}: '
            f'Reward={self.episode_reward:.2f}, '
            f'Steps={self.episode_steps}'
        )
        
        if self.episodes_completed % 10 == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(
                self.save_path,
                f'{self.model_type}_episode_{self.episodes_completed}_{timestamp}.pt'
            )
            self.agent.save(save_path)
            self.get_logger().info(f'Saved model to {save_path}')
        
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.pending_transition = None


def main(args=None):
    rclpy.init(args=args)
    node = RLAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
