import time
import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import HerReplayBuffer, SAC, TD3


gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v2")
# Create 4 artificial transitions per real transition
n_sampled_goal = 16

# model = SAC(
#     "MultiInputPolicy",
#     env,
#     replay_buffer_class=HerReplayBuffer,
#     replay_buffer_kwargs=dict(
#         n_sampled_goal=n_sampled_goal,
#         goal_selection_strategy="future",
#     ),
#     verbose=1,
#     buffer_size=int(1e6),
#     learning_rate=1e-3,
#     gamma=0.99,
#     batch_size=256,
#     policy_kwargs=dict(net_arch=[256, 256]),
#     tensorboard_log="./logs/"
# )
# model.learn(int(2e4))
# model.save("her_sac_highway")

# Load saved model
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
env = gym.make("FetchReach-v2", render_mode="human") # Change the render mode
model = SAC.load("her_sac_highway", env=env)

obs, info = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(1000):
    # time.sleep(0.04)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    if terminated or truncated or info.get("is_success", False):
        print("Reward:", episode_reward, "Success?", info.get("is_success", False))
        episode_reward = 0.0
        obs, info = env.reset()
