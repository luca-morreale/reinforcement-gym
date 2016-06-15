# -*- coding: utf-8 -*-
from mc_policy import MCPolicy  # lint:ok
from td_policy import TDPolicy  # lint:ok
from sarsa_policy import SarsaPolicy  # lint:ok
from epsilon_greedy import EpsilonGreedyChooser
import gym
#error somewhere, check comparision of hash, equalitiy etc...


def main():
    env = gym.make('CartPole-v0')
    epsilon = 0.99
    alfa = 0.6
    discount_factor = 1
    cellSize = 0.0009

    # def __init__(self, discount_factor, learning_rate, epsilon):

    action_chooser = EpsilonGreedyChooser(epsilon, env.action_space.n)

    #pi = MCPolicy(action_chooser, discount_factor, alfa)
    #pi.set(env, cellSize)

    #env.monitor.start('./cartpole-experiment-1')

    #for episode in range(1, 400):
    #    pi.doEpisode(episode)

    #env.monitor.close()

    print()

    pi2 = SarsaPolicy(action_chooser, discount_factor, alfa)
    pi2.set(env, cellSize)

    #env.monitor.start('./cartpole-experiment-1')

    for episode in range(1, 600):
        pi2.doEpisode(episode)

    #env.monitor.close()


if __name__ == "__main__":
    main()