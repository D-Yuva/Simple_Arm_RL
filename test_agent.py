import numpy as np
from stable_baselines3 import PPO
from simple_arm_env.simple_arm import SimpleArmEnv  # import your env class

env = SimpleArmEnv(render=True)  # enable GUI

model = PPO.load("ppo_simple_arm.zip")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
