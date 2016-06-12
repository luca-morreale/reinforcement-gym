# -*- coding: utf-8 -*-
from mc_policy import MCPolicy  # lint:ok
from td_policy import TDPolicy  # lint:ok
import gym


def main():
    env = gym.make('CartPole-v0')
    epsilon = 0.99
    alfa = 0.1
    discount_factor = 0.9

    # def __init__(self, discount_factor, learning_rate, epsilon):

    pi2 = TDPolicy(discount_factor, alfa, epsilon)
    pi2.set(env)

    #env.monitor.start('./cartpole-experiment-1')

    for episode in range(1, 15000):
        pi2.doEpisode(episode)

    #env.monitor.close()


if __name__ == "__main__":
    main()