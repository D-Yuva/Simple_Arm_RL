import gymnasium as gym  # ✅ not 'gym'
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import os

class SimpleArmEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()

        self.render_mode = render
        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        urdf_path = os.path.join(os.path.dirname(__file__), "arm.urdf")
        self.arm = p.loadURDF(urdf_path, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.arm)

        # Observation space: joint positions and velocities
        obs_high = np.array([np.pi] * self.num_joints + [100.0] * self.num_joints, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Action space: 3 continuous actions scaled to [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)

        # Example fixed target position (3D)
        self.target_pos = np.array([0.3, 0.0, 0.3])

        # Define a fixed target in 3D space or randomize on reset
        self.target_pos = np.array([0.3, 0.0, 0.3])  # reachable target

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for i in range(self.num_joints):
            p.resetJointState(self.arm, i, targetValue=0)
        
        # Optionally randomize target here for variety
        self.target_pos = np.array([0.3, 0.0, 0.3])
        
        return self._get_obs(), {}

    def step(self, action):
        # Clip actions to joint limits scaled from [-1,1]
        # Convert your action space (-1,1) to actual joint angle limits
        scaled_action = np.clip(action, -1, 1)
        joint_limits = [(-3.14, 3.14), (-1.57, 1.57), (0, 2.0)]
        target_positions = []
        for i, (low, high) in enumerate(joint_limits):
            # scale from [-1,1] to [low, high]
            target_positions.append(low + (scaled_action[i] + 1) * 0.5 * (high - low))

        for i in range(self.num_joints):
            p.setJointMotorControl2(
                bodyIndex=self.arm,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                force=20,
                positionGain=0.1,
                velocityGain=1.0
            )
        p.stepSimulation()
        if self.render_mode:
            time.sleep(1. / 240.)

        obs = self._get_obs()
        
        hand_pos = self.get_hand_pos()
        dist = np.linalg.norm(hand_pos - self.target_pos)
        
        # Reward = negative distance + small penalty for joint velocity (optional)
        reward = -dist - 0.01 * np.linalg.norm(obs[self.num_joints:])  
        
        # Episode done when close enough
        done = bool(dist < 0.05)
        
        return obs, reward, done, False, {}

    def get_hand_pos(self):
        pos, _ = p.getLinkState(self.arm, 2)[:2]
        return np.array(pos)
    
    def _get_obs(self):
        joint_positions = []
        joint_velocities = []
        for i in range(self.num_joints):
            joint_state = p.getJointState(self.arm, i)
            joint_positions.append(joint_state[0])  # position
            joint_velocities.append(joint_state[1])  # velocity
        
        obs = np.array(joint_positions + joint_velocities, dtype=np.float32)
        return obs
