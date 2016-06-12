# MsPacman-v0
import time
import gym
from gym import envs

env = gym.make('CartPole-v0')
print((envs.registration.EnvSpec('CartPole-v0')))

print((env.action_space))  # = Discrete(2) = 2 type of actions
print((env.observation_space))  # = Box(4,) = array of length 4


for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(("Episode finished after {} timesteps".format(t + 1)))
            break