# -*- coding: utf-8 -*-
from policy.sarsa_policy import SarsaPolicy

from action_chooser.epsilon_greedy_chooser import EpsilonGreedyChooser
from action_chooser.variation_epsilon_chooser import VariationEpsilonGreedyChooser
from action_chooser.softmax_chooser import SoftmaxChooser


from generalizer.tiles_state_generalizer import TilesStateGeneralizer

from updater.updater import Updater
from updater.trace_updater import UpdaterTraced

import gym


#MountainCar-v0
#CartPole-v0

def main():
    env = gym.make('CartPole-v0')
    cellSize = 2

    num_tilings = 10
    num_tiles = 8
    obs_space = env.observation_space
    m = env.action_space.n
    n = 30000

    epsilon = 0.1

    alfa = 0.5
    gamma = 1.0
    lambda_ = 0.9

    action_chooser = EpsilonGreedyChooser(epsilon, env.action_space.n)
    action_chooser = VariationEpsilonGreedyChooser(epsilon, env)

    generalizer = TilesStateGeneralizer(num_tilings, num_tiles, obs_space, m, n)

    #updater = Updater(discount_factor, alfa, generalizer)
    updater = UpdaterTraced(gamma, alfa, generalizer, lambda_)

    pi = SarsaPolicy(action_chooser, generalizer, updater)
    pi.set(env, cellSize)

    #env.monitor.start('./cartpole-experiment-1')

    for episode in range(1, 2000):
        pi.doEpisode(episode)

    #env.monitor.close()
    #generalizer.prettyPrintQ()


if __name__ == "__main__":
    main()