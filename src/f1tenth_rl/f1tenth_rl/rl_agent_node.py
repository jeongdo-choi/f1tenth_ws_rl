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


SB3_MODEL_TYPES = ('sb3_ppo', 'stable_baselines3_ppo')


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
        self.declare_parameter('speed_odom_topic', '/odom')
        self.declare_parameter('drive_steering_limit', 0.4)
        self.declare_parameter('drive_min_speed', 0.0)
        self.declare_parameter('drive_max_speed', 1.0)
        self.declare_parameter('sb3_scan_beams', 2155)
        self.declare_parameter('sb3_lidar_max_range', 10.0)
        self.declare_parameter('sb3_speed_scale', 3.2)
        self.declare_parameter('sb3_action_steering_limit', 0.4189)
        self.declare_parameter('sb3_action_min_speed', 0.5)
        self.declare_parameter('sb3_action_max_speed', 4.0)
        
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
        self.speed_odom_topic = self.get_parameter('speed_odom_topic').value
        self.drive_steering_limit = float(self.get_parameter('drive_steering_limit').value)
        self.drive_min_speed = float(self.get_parameter('drive_min_speed').value)
        self.drive_max_speed = float(self.get_parameter('drive_max_speed').value)
        self.sb3_scan_beams = int(self.get_parameter('sb3_scan_beams').value)
        self.sb3_lidar_max_range = float(self.get_parameter('sb3_lidar_max_range').value)
        self.sb3_speed_scale = float(self.get_parameter('sb3_speed_scale').value)
        self.sb3_action_steering_limit = float(self.get_parameter('sb3_action_steering_limit').value)
        self.sb3_action_min_speed = float(self.get_parameter('sb3_action_min_speed').value)
        self.sb3_action_max_speed = float(self.get_parameter('sb3_action_max_speed').value)
        
        # Create directories if they don't exist
        os.makedirs(self.save_path, exist_ok=True)
        
        # Store the latest observations
        self.latest_scan = None
        self.latest_odom = None
        self.prev_odom = None
        self.latest_speed = 0.0
        self.has_speed_odom = False
        self.warned_missing_speed = False
        
        # Publishers and subscribers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/pf/pose/odom', self.odom_callback, 10)

        self.speed_odom_sub = self.create_subscription(
            Odometry, self.speed_odom_topic, self.speed_odom_callback, 10)
        
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
        elif self.model_type in SB3_MODEL_TYPES:
            if self.training_mode:
                raise ValueError('SB3 PPO mode is for deployment only. Launch with training_mode:=false.')
            if not self.model_path:
                raise ValueError('SB3 PPO mode requires model_path to point to a Stable-Baselines3 .zip model.')

            inferred_state_dim = self._infer_sb3_state_dim(self.model_path)
            if inferred_state_dim is not None and inferred_state_dim != self.state_dim:
                self.get_logger().warn(
                    f'Overriding state_dim from {self.state_dim} to {inferred_state_dim} '
                    'to match the SB3 PPO checkpoint.'
                )
                self.state_dim = inferred_state_dim

            from stable_baselines3 import PPO as SB3PPO
            custom_objects = self._make_sb3_custom_objects()
            self.agent = SB3PPO.load(self.model_path, custom_objects=custom_objects)
            self.actions = None
            self.get_logger().info(f'Loaded Stable-Baselines3 PPO model from {self.model_path}')
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        # Load custom PyTorch checkpoints if provided.
        if self.model_path and self.model_type not in SB3_MODEL_TYPES:
            try:
                self.agent.load(self.model_path)
                self.get_logger().info(f'Loaded model from {self.model_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to load model: {e}')

    def _infer_sb3_state_dim(self, model_path):
        """Infer the flat observation size from a Stable-Baselines3 .zip policy."""
        import io
        import zipfile

        try:
            with zipfile.ZipFile(model_path, 'r') as model_zip:
                policy_bytes = model_zip.read('policy.pth')
        except Exception as e:
            self.get_logger().warn(f'Could not inspect SB3 policy.pth: {e}')
            return None

        try:
            policy_state = torch.load(io.BytesIO(policy_bytes), map_location='cpu')
        except Exception as e:
            self.get_logger().warn(f'Could not load SB3 policy state for inspection: {e}')
            return None

        for key, value in policy_state.items():
            if key.endswith('mlp_extractor.policy_net.0.weight') and len(value.shape) == 2:
                return int(value.shape[1])

        for key, value in policy_state.items():
            if key.endswith('features_extractor.flatten.weight') and len(value.shape) >= 2:
                return int(value.shape[1])

        self.get_logger().warn('Could not infer SB3 observation size from policy weights.')
        return None

    def _make_sb3_custom_objects(self):
        """Provide spaces explicitly when old SB3 pickles cannot deserialize them."""
        try:
            from gymnasium import spaces
        except ImportError:
            from gym import spaces

        observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        return {
            'observation_space': observation_space,
            'action_space': action_space,
        }
    
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

    def speed_odom_callback(self, msg):
        """Store speed for SB3 sim-to-real observations."""
        self.latest_speed = float(msg.twist.twist.linear.x)
        self.has_speed_odom = True
    
    def get_state(self):
        """Convert laser scan to a fixed-size state vector for the RL agent"""
        if self.latest_scan is None:
            return None

        if self.model_type in SB3_MODEL_TYPES:
            return self._get_sb3_state()

        return self._get_scan_state(self.state_dim)

    def _get_scan_state(self, target_size):
        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(
            ranges,
            nan=self.sb3_lidar_max_range,
            posinf=self.sb3_lidar_max_range,
            neginf=0.0
        )
        ranges = np.clip(ranges, 0.0, self.sb3_lidar_max_range)
        normalized_ranges = ranges / self.sb3_lidar_max_range

        if len(normalized_ranges) == target_size:
            return normalized_ranges.astype(np.float32)

        if len(normalized_ranges) == 0:
            return np.ones(target_size, dtype=np.float32)

        source_angles = np.linspace(
            self.latest_scan.angle_min,
            self.latest_scan.angle_max,
            len(normalized_ranges),
            dtype=np.float32
        )
        target_angles = np.linspace(
            self.latest_scan.angle_min,
            self.latest_scan.angle_max,
            target_size,
            dtype=np.float32
        )
        return np.interp(target_angles, source_angles, normalized_ranges).astype(np.float32)

    def _get_sb3_state(self):
        scan_state = self._get_scan_state(self.sb3_scan_beams)
        speed_feature = self._get_speed_feature()

        if self.state_dim == self.sb3_scan_beams:
            return scan_state

        state = np.concatenate(
            [scan_state, np.array([speed_feature], dtype=np.float32)]
        )

        if len(state) > self.state_dim:
            state = state[:self.state_dim]
        elif len(state) < self.state_dim:
            state = np.pad(
                state,
                (0, self.state_dim - len(state)),
                mode='constant',
                constant_values=1.0
            )

        return state.astype(np.float32)

    def _get_speed_feature(self):
        if not self.has_speed_odom and not self.warned_missing_speed:
            self.get_logger().warn(
                f'No odometry received on {self.speed_odom_topic}; using speed feature 0.0.'
            )
            self.warned_missing_speed = True

        if self.sb3_speed_scale <= 0.0:
            return 0.0

        return float(np.clip(self.latest_speed / self.sb3_speed_scale, 0.0, 1.0))
    
    def publish_drive_command(self, steering, velocity):
        """Publish drive command to the car"""
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(
            np.clip(steering, -self.drive_steering_limit, self.drive_steering_limit)
        )
        msg.drive.speed = float(
            np.clip(velocity, self.drive_min_speed, self.drive_max_speed)
        )
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

        if self.model_type in SB3_MODEL_TYPES:
            action, _ = self.agent.predict(state, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size < 2:
                raise ValueError(f'SB3 PPO action must contain steering and velocity, got shape {action.shape}')
            steering, velocity = self._convert_sb3_action(action)
            return steering, velocity, {'raw_action': action}

        deterministic = not self.training_mode
        action, log_prob, value = self.agent.select_action(state, deterministic=deterministic)
        steering, velocity = action
        return steering, velocity, {
            'action': action,
            'log_prob': log_prob,
            'value': value,
        }

    def _convert_sb3_action(self, action):
        steering_norm = float(np.clip(action[0], -1.0, 1.0))
        speed_norm = float(np.clip(action[1], -1.0, 1.0))
        speed_ratio = (speed_norm + 1.0) * 0.5

        steering = steering_norm * self.sb3_action_steering_limit
        velocity = self.sb3_action_min_speed + speed_ratio * (
            self.sb3_action_max_speed - self.sb3_action_min_speed
        )
        velocity = float(np.clip(velocity, self.drive_min_speed, self.drive_max_speed))
        return steering, velocity
    
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