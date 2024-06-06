import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1')
env.reset()
for _ in range(1000):
    env.render()
    # 采取随机策略
    env.step(env.action_space.sample())

env.close()

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1
totals = []
for episode in range(50):
    episode_rewards = 0
    obs, info = env.reset()
    # env.render()
    # print(obs)
    for step in range(100):
        action = basic_policy(obs)
        obs, reward, mdp_done, done, info = env.step(action)
        episode_rewards += reward
        if mdp_done or done:
            break
    totals.append(episode_rewards)
    # print(np.max(totals))

env.close()

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))