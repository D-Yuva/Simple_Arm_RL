from stable_baselines3 import PPO
from simple_arm_env.simple_arm import SimpleArmEnv
from stable_baselines3.common.env_checker import check_env

env = SimpleArmEnv(render=False)

check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_simple_arm")
