import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from IPython import display
import gymnasium as gym


def dis_to_con(action_id, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    return action_lowbound + (action_id / (action_dim - 1)) * (
        action_upbound - action_lowbound
    )

def random_run():
    # The episode truncates at 200 time steps.
    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)  # default g=10.0

    action_dim = 11
    obs_dim = env.observation_space.shape[0]
    obs_dim

    trajs = []
    traj_num = int(1e3)
    for i in range(traj_num):
        cur_traj = []
        cur_obs, _ = env.reset(options={"x_init": np.pi / 2, "y_init": 0})
        print(f"current traj num : {i} , current obs : {cur_obs}")
        done = False
        trunk = False
        while not (done or trunk):
            action_id = np.random.choice(action_dim)
            action = dis_to_con(action_id, env, action_dim)
            obs, reward, done, trunk, _ = env.step([action.astype(np.float64)])
            cur_traj.append(np.concatenate([cur_obs, [action_id], obs]))
            cur_obs = obs
            env.render()
            if done or trunk:
                print(done, trunk)
                trajs.append(cur_traj)

data = np.load("./QVW.npz")
Q = data['arr_0']
V = data['arr_1']
W = data['arr_2']

env = gym.make("Pendulum-v1", render_mode="human")
# env = gym.make("CartPole-v1", render_mode="human")
# action_dim = 11

def CML_controller(start, goal):
    delta = Q@goal - Q@start
    utility = delta.T@V
    best_id = np.argmax(utility)
    # action = dis_to_con(best_id, env, action_dim)
    return best_id

import os

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
rewards = []
for _ in range(100):
    cur_obs, _ = env.reset()#(options={"x_init": 0, "y_init": 0})
    done = False
    trunk = False
    episode_reward = 0
    while not (done or trunk):
        goal = cur_obs.copy()
        goal[0] = 0
        goal[1] = 1
        goal[2] = 0
        # goal[0] = 0
        # goal[1:] = 0
        # print(cur_obs, goal)
        best_id = CML_controller(cur_obs, goal)
        best_id = env.action_space.sample()
        # print(np.arctan2(cur_obs[1], cur_obs[0]) * 180 / np.pi, cur_obs[:2])
        cur_obs, reward, done, trunk, _ = env.step(best_id)
        episode_reward += reward
        # env.render()
        if done or trunk:
            # print(done, trunk)
            rewards.append(np.mean(episode_reward))
plt.plot(rewards, label="random action")
rewards = []
for _ in range(100):
    cur_obs, _ = env.reset(options={"x_init": 0, "y_init": 0})
    done = False
    trunk = False
    episode_reward = 0
    while not (done or trunk):
        goal = cur_obs.copy()
        # goal[0] = 0
        goal[1:] = 0
        # print(cur_obs, goal)
        best_id = CML_controller(cur_obs, goal)
        # best_id = env.action_space.sample()
        # print(np.arctan2(cur_obs[1], cur_obs[0]) * 180 / np.pi, cur_obs[:2])
        cur_obs, reward, done, trunk, _ = env.step(best_id)
        episode_reward += reward
        # env.render()
        if done or trunk:
            # print(done, trunk)
            rewards.append(np.mean(episode_reward))
plt.plot(rewards, label="CML discrete action")
plt.legend()
plt.show()