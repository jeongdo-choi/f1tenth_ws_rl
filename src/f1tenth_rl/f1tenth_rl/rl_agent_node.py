#!/usr/bin/env python3

import json
import os
from datetime import datetime

import numpy as np
import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

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
        self.declare_parameter('drive_steering_smoothing_alpha', 0.45)
        self.declare_parameter('drive_max_steering_delta', 0.08)
        self.declare_parameter('drive_turn_speed_reduction', 0.25)
        self.declare_parameter('sb3_observation_layout', 'auto')
        self.declare_parameter('sb3_scan_beams', 2155)
        self.declare_parameter('sb3_lidar_max_range', 10.0)
        self.declare_parameter('sb3_reverse_scan', False)
        self.declare_parameter('sb3_scan_speed_order', 'speed_first')
        self.declare_parameter('sb3_speed_scale', 3.2)
        self.declare_parameter('sb3_default_pose_d', 0.0)
        self.declare_parameter('sb3_centerline_csv', '')
        self.declare_parameter('sb3_centerline_x_col', 0)
        self.declare_parameter('sb3_centerline_y_col', 1)
        self.declare_parameter('sb3_centerline_yaw_col', 2)
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
        self.drive_steering_smoothing_alpha = float(np.clip(
            self.get_parameter('drive_steering_smoothing_alpha').value,
            0.0,
            1.0
        ))
        self.drive_max_steering_delta = float(
            self.get_parameter('drive_max_steering_delta').value
        )
        self.drive_turn_speed_reduction = float(np.clip(
            self.get_parameter('drive_turn_speed_reduction').value,
            0.0,
            1.0
        ))
        self.sb3_observation_layout = self.get_parameter('sb3_observation_layout').value
        self.sb3_scan_beams = int(self.get_parameter('sb3_scan_beams').value)
        self.sb3_lidar_max_range = float(self.get_parameter('sb3_lidar_max_range').value)
        self.sb3_reverse_scan = bool(self.get_parameter('sb3_reverse_scan').value)
        self.sb3_scan_speed_order = str(
            self.get_parameter('sb3_scan_speed_order').value
        ).lower().strip()
        self.sb3_speed_scale = float(self.get_parameter('sb3_speed_scale').value)
        self.sb3_default_pose_d = float(self.get_parameter('sb3_default_pose_d').value)
        self.sb3_centerline_csv = self.get_parameter('sb3_centerline_csv').value
        self.sb3_centerline_x_col = int(self.get_parameter('sb3_centerline_x_col').value)
        self.sb3_centerline_y_col = int(self.get_parameter('sb3_centerline_y_col').value)
        self.sb3_centerline_yaw_col = int(self.get_parameter('sb3_centerline_yaw_col').value)
        self.sb3_action_steering_limit = float(self.get_parameter('sb3_action_steering_limit').value)
        self.sb3_action_min_speed = float(self.get_parameter('sb3_action_min_speed').value)
        self.sb3_action_max_speed = float(self.get_parameter('sb3_action_max_speed').value)
        self.sb3_action_low = np.array([-1.0, -1.0], dtype=np.float32)
        self.sb3_action_high = np.array([1.0, 1.0], dtype=np.float32)

        if self.sb3_scan_speed_order not in ('speed_first', 'scan_first'):
            self.get_logger().warn(
                f'Unknown sb3_scan_speed_order={self.sb3_scan_speed_order}; using speed_first.'
            )
            self.sb3_scan_speed_order = 'speed_first'

        # Create directories if they don't exist
        os.makedirs(self.save_path, exist_ok=True)

        # Store the latest observations
        self.latest_scan = None
        self.latest_odom = None
        self.prev_odom = None
        self.latest_speed = 0.0
        self.latest_linear_vel_s = 0.0
        self.latest_ang_vel_z = 0.0
        self.latest_yaw = 0.0
        self.latest_pose_d = self.sb3_default_pose_d
        self.has_speed_odom = False
        self.warned_missing_speed = False
        self.warned_frenet_approximation = False
        self.centerline_points = None
        self.centerline_yaws = None
        self.last_steering_cmd = 0.0

        # Publishers and subscribers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/pf/pose/odom', self.odom_callback, 10)

        self.speed_odom_sub = self.create_subscription(
            Odometry, self.speed_odom_topic, self.speed_odom_callback, 10)

        self._load_centerline()

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

            metadata = self._inspect_sb3_metadata(self.model_path)
            inferred_state_dim = metadata.get('state_dim') or self._infer_sb3_state_dim(self.model_path)
            if inferred_state_dim is not None and inferred_state_dim != self.state_dim:
                self.get_logger().warn(
                    f'Overriding state_dim from {self.state_dim} to {inferred_state_dim} '
                    'to match the SB3 PPO checkpoint.'
                )
                self.state_dim = inferred_state_dim

            if metadata.get('action_low') is not None and metadata.get('action_high') is not None:
                self.sb3_action_low = metadata['action_low']
                self.sb3_action_high = metadata['action_high']

            from stable_baselines3 import PPO as SB3PPO
            custom_objects = self._make_sb3_custom_objects()
            self.agent = SB3PPO.load(self.model_path, custom_objects=custom_objects)
            self.actions = None
            self.get_logger().info(f'Loaded Stable-Baselines3 PPO model from {self.model_path}')
            self.get_logger().info(
                f'SB3 observation_dim={self.state_dim}, '
                f'action_low={self.sb3_action_low.tolist()}, '
                f'action_high={self.sb3_action_high.tolist()}, '
                f'observation_layout={self._resolve_sb3_observation_layout()}, '
                f'reverse_scan={self.sb3_reverse_scan}, '
                f'scan_speed_order={self.sb3_scan_speed_order}, '
                f'steering_smoothing_alpha={self.drive_steering_smoothing_alpha}, '
                f'max_steering_delta={self.drive_max_steering_delta}, '
                f'turn_speed_reduction={self.drive_turn_speed_reduction}'
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # Load custom PyTorch checkpoints if provided.
        if self.model_path and self.model_type not in SB3_MODEL_TYPES:
            try:
                self.agent.load(self.model_path)
                self.get_logger().info(f'Loaded model from {self.model_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to load model: {e}')

    def _inspect_sb3_metadata(self, model_path):
        """Read lightweight SB3 metadata from the zip without unpickling objects."""
        import zipfile

        metadata = {'state_dim': None, 'action_low': None, 'action_high': None}
        try:
            with zipfile.ZipFile(model_path, 'r') as model_zip:
                data = json.loads(model_zip.read('data').decode('utf-8'))
        except Exception as e:
            self.get_logger().warn(f'Could not inspect SB3 data metadata: {e}')
            return metadata

        observation_space = data.get('observation_space', {})
        shape = observation_space.get('shape')
        if isinstance(shape, list) and shape:
            metadata['state_dim'] = int(np.prod(shape))

        action_space = data.get('action_space', {})
        action_low = self._parse_space_vector(action_space.get('low'), expected_size=2)
        action_high = self._parse_space_vector(action_space.get('high'), expected_size=2)
        if action_low is not None and action_high is not None:
            metadata['action_low'] = action_low
            metadata['action_high'] = action_high

        return metadata

    def _parse_space_vector(self, value, expected_size):
        if value is None:
            return None

        if isinstance(value, list):
            parsed = np.asarray(value, dtype=np.float32)
        else:
            cleaned = str(value).replace('[', ' ').replace(']', ' ').replace(',', ' ')
            parsed = np.fromstring(cleaned, sep=' ', dtype=np.float32)

        if parsed.size != expected_size or not np.all(np.isfinite(parsed)):
            return None

        return parsed.astype(np.float32)

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
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        action_space = spaces.Box(
            low=self.sb3_action_low.astype(np.float32),
            high=self.sb3_action_high.astype(np.float32),
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
        """Store speed and Frenet-like motion features for SB3 observations."""
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = self._yaw_from_quaternion(msg.pose.pose.orientation)
        speed = float(msg.twist.twist.linear.x)

        self.latest_speed = speed
        self.latest_linear_vel_s = speed
        self.latest_ang_vel_z = float(msg.twist.twist.angular.z)
        self.latest_yaw = yaw
        self.has_speed_odom = True
        self._update_centerline_projection(x, y, yaw, speed)

    def _yaw_from_quaternion(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _load_centerline(self):
        path = str(self.sb3_centerline_csv).strip()
        if not path:
            self.get_logger().warn(
                'No sb3_centerline_csv configured; 2160-dim SB3 poses_d will use sb3_default_pose_d.'
            )
            return

        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        if not os.path.exists(path):
            self.get_logger().warn(f'SB3 centerline CSV not found: {path}')
            return

        try:
            data = np.genfromtxt(path, delimiter=',', comments='#', dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f'Could not load SB3 centerline CSV {path}: {e}')
            return

        if data.ndim == 1:
            data = data.reshape(1, -1)

        max_col = max(self.sb3_centerline_x_col, self.sb3_centerline_y_col, self.sb3_centerline_yaw_col)
        if data.ndim != 2 or data.shape[1] <= max_col:
            self.get_logger().warn(
                f'SB3 centerline CSV {path} does not have required columns up to {max_col}.'
            )
            return

        points = data[:, [self.sb3_centerline_x_col, self.sb3_centerline_y_col]]
        yaws = data[:, self.sb3_centerline_yaw_col]
        finite_mask = np.isfinite(points).all(axis=1) & np.isfinite(yaws)
        points = points[finite_mask]
        yaws = yaws[finite_mask]

        if len(points) < 2:
            self.get_logger().warn(f'SB3 centerline CSV {path} needs at least two valid points.')
            return

        self.centerline_points = points.astype(np.float32)
        self.centerline_yaws = yaws.astype(np.float32)
        self.get_logger().info(f'Loaded {len(points)} SB3 centerline points from {path}')

    def _update_centerline_projection(self, x, y, yaw, speed):
        if self.centerline_points is None or self.centerline_yaws is None:
            self.latest_pose_d = self.sb3_default_pose_d
            self.latest_linear_vel_s = speed
            return

        current = np.array([x, y], dtype=np.float32)
        deltas = self.centerline_points - current
        nearest_idx = int(np.argmin(np.einsum('ij,ij->i', deltas, deltas)))
        closest_x, closest_y = self.centerline_points[nearest_idx]
        centerline_yaw = float(self.centerline_yaws[nearest_idx])

        dx = x - float(closest_x)
        dy = y - float(closest_y)
        self.latest_pose_d = float(dx * np.cos(centerline_yaw) + dy * np.sin(centerline_yaw))

        # This mirrors rldd's FrenetObsWrapper convention for linear_vels_s.
        self.latest_linear_vel_s = float(speed * np.sin(yaw - centerline_yaw))

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
        if self.sb3_reverse_scan:
            normalized_ranges = normalized_ranges[::-1]

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
        layout = self._resolve_sb3_observation_layout()

        if layout == 'rldd_frenet_scan':
            return self._get_sb3_frenet_scan_state(scan_state)

        speed_feature = np.array([self._get_speed_feature()], dtype=np.float32)
        if self.sb3_scan_speed_order == 'scan_first':
            state = np.concatenate([scan_state, speed_feature])
        else:
            state = np.concatenate([speed_feature, scan_state])
        return self._match_state_dim(state)

    def _resolve_sb3_observation_layout(self):
        layout = str(self.sb3_observation_layout).lower().strip()
        if layout != 'auto':
            return layout

        if self.state_dim == self.sb3_scan_beams + 5:
            return 'rldd_frenet_scan'

        return 'scan_speed'

    def _get_sb3_frenet_scan_state(self, scan_state):
        if self.centerline_points is None and not self.warned_frenet_approximation:
            self.get_logger().warn(
                'Using rldd 2160 observation layout without a centerline; '
                'poses_d defaults to sb3_default_pose_d.'
            )
            self.warned_frenet_approximation = True

        features = np.array([
            self.latest_ang_vel_z,
            self._get_speed_feature(),
            self.latest_linear_vel_s,
            self.latest_pose_d,
            self.latest_yaw,
        ], dtype=np.float32)
        return self._match_state_dim(np.concatenate([features, scan_state]))

    def _match_state_dim(self, state):
        if len(state) > self.state_dim:
            return state[:self.state_dim].astype(np.float32)

        if len(state) < self.state_dim:
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

    def _smooth_drive_command(self, steering, velocity):
        target_steering = float(
            np.clip(steering, -self.drive_steering_limit, self.drive_steering_limit)
        )

        if 0.0 < self.drive_steering_smoothing_alpha < 1.0:
            target_steering = self.last_steering_cmd + self.drive_steering_smoothing_alpha * (
                target_steering - self.last_steering_cmd
            )

        if self.drive_max_steering_delta > 0.0:
            steering_delta = float(np.clip(
                target_steering - self.last_steering_cmd,
                -self.drive_max_steering_delta,
                self.drive_max_steering_delta
            ))
            target_steering = self.last_steering_cmd + steering_delta

        target_steering = float(
            np.clip(target_steering, -self.drive_steering_limit, self.drive_steering_limit)
        )
        self.last_steering_cmd = target_steering

        target_speed = float(
            np.clip(velocity, self.drive_min_speed, self.drive_max_speed)
        )
        if self.drive_turn_speed_reduction > 0.0 and self.drive_steering_limit > 0.0:
            turn_ratio = min(abs(target_steering) / self.drive_steering_limit, 1.0)
            target_speed *= 1.0 - self.drive_turn_speed_reduction * turn_ratio
            target_speed = float(
                np.clip(target_speed, self.drive_min_speed, self.drive_max_speed)
            )

        return target_steering, target_speed

    def publish_drive_command(self, steering, velocity):
        """Publish drive command to the car"""
        steering, velocity = self._smooth_drive_command(steering, velocity)
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering
        msg.drive.speed = velocity
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
        steering_low = float(self.sb3_action_low[0])
        steering_high = float(self.sb3_action_high[0])
        speed_low = float(self.sb3_action_low[1])
        speed_high = float(self.sb3_action_high[1])

        steering_norm = float(np.clip(action[0], steering_low, steering_high))
        speed_norm = float(np.clip(action[1], speed_low, speed_high))

        if steering_high != steering_low:
            steering_ratio = (steering_norm - steering_low) / (steering_high - steering_low)
            steering_norm = steering_ratio * 2.0 - 1.0

        if speed_high == speed_low:
            speed_ratio = 0.0
        else:
            speed_ratio = (speed_norm - speed_low) / (speed_high - speed_low)

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
        self.last_steering_cmd = 0.0
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
