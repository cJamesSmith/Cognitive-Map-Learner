import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
def train():
    vec_env = make_vec_env("Pendulum-v1", n_envs=8, seed=0)
    vec_env = make_vec_env("CartPole-v1", n_envs=8, seed=0)

    # We collect 4 transitions per call to `env.step()`
    # and performs 2 gradient steps per call to `env.step()`
    # if gradient_steps=-1, then we would do 4 gradients steps per call to `env.step()`
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=200000)
    model.save("PPO_CartPole")
    
def test():
    env = gym.make("Pendulum-v1", render_mode='human')
    env = gym.make("CartPole-v1", render_mode=None)
    model = PPO.load("PPO_CartPole", env=env)
    rewards = []
    # Enjoy trained test\
    for _ in range(100):
        cur_obs, _ = env.reset(options={"x_init": np.pi, "y_init": 1})
        done = False
        trunk = False
        episode_reward = 0
        while not (done or trunk):
            best_id, _ = model.predict(cur_obs)
            # print(np.arctan2(cur_obs[1], cur_obs[0]) * 180 / np.pi, cur_obs[:2])
            cur_obs, reward, done, trunk, _ = env.step(best_id)
            episode_reward += reward
            # env.render()
            if done or trunk:
                print(done, trunk)
                rewards.append(np.mean(episode_reward))
    plt.plot(rewards, label="PPO discrete action")
    plt.legend()
    plt.show()
        
if __name__ == '__main__':
    test()