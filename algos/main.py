# -*- coding: utf-8 -*-
from policy.mc_policy import MCPolicy  # lint:ok
from policy.td_policy import TDPolicy  # lint:ok
from policy.sarsa_policy import SarsaPolicy  # lint:ok

from action_chooser.epsilon_greedy_chooser import EpsilonGreedyChooser
from action_chooser.softmax_chooser import SoftmaxChooser

from generalizer.hash_state_generalizer import HashGeneralizer
from generalizer.norm_state_generalizer import NormGeneralizer

from updater.updater import Updater
from updater.trace_updater import UpdaterTraced

import gym


#MountainCar-v0
#CartPole-v0

def main():
    env = gym.make('CartPole-v0')
    epsilon = 0.1
    alfa = 0.5
    discount_factor = 1
    lambda_ = 0.9
    cellSize = 0.0002
    temperature = 5

    #action_chooser = SoftmaxChooser(temperature)
    action_chooser = EpsilonGreedyChooser(epsilon, env.action_space.n)

    #generalizer = NormGeneralizer(cellSize)
    generalizer = HashGeneralizer(cellSize)

    #updater = Updater(discount_factor, alfa)
    updater = UpdaterTraced(discount_factor, alfa, lambda_)

    #pi = MCPolicy(action_chooser, generalizer, updater)
    #pi = TDPolicy(action_chooser, generalizer, updater)
    pi = SarsaPolicy(action_chooser, generalizer, updater)
    pi.set(env, cellSize)

    #env.monitor.start('./cartpole-experiment-1')

    for episode in range(1, 5000):
        pi.doEpisode(episode)

    #env.monitor.close()


if __name__ == "__main__":
    main()