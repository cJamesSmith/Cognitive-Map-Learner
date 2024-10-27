import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import HerReplayBuffer, SAC, TD3


gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v2")
print(env.reset())
print(env.action_space.sample)
