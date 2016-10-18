# -*- coding: utf-8 -*-
import gym
from generalizer.coding.tiles_state_generalizer import TilesStateGeneralizer

from action_chooser.epsilon_greedy_chooser import EpsilonGreedyChooser
from policy.sarsa_policy import SarsaPolicy
from updater.trace_updater import UpdaterTraced


#MountainCar-v0
#CartPole-v0
#Pendulum-v0
#Acrobot-v1

def main():
    env = gym.make('Pendulum-v0')
    cellSize = 0.5
    num_tilings = 10
    num_tiles = 3
    obs_space = env.observation_space
    print(env.action_space.high)
    print(env.action_space.low)
    m = env.action_space.n
    n = 30000

    epsilon = 0.1

    alfa = 0.5
    gamma = 1.0
    lambda_ = 0.9

    action_chooser = EpsilonGreedyChooser(epsilon, env.action_space.n)

    #generalizer = NoneGeneralizer(m)
    #generalizer = NormGeneralizer(m, cellSize)
    #generalizer = HashGeneralizer(m, cellSize, obs_space)
    generalizer = TilesStateGeneralizer(num_tilings, num_tiles, obs_space, m, n)

    #updater = Updater(gamma, alfa, generalizer)
    updater = UpdaterTraced(gamma, alfa, generalizer, lambda_)

    #pi = MCPolicy(action_chooser, generalizer, updater)
    #pi = TDPolicy(action_chooser, generalizer, updater)
    pi = SarsaPolicy(action_chooser, generalizer, updater)
    #pi = QLearningPolicy(generalizer, updater, m, 50)
    pi.set(env)

    #env.monitor.start('./cartpole-experiment-1')

    for episode in range(1, 1000):
        pi.doEpisode(episode)

    #env.monitor.close()
    #generalizer.prettyPrintQ()


if __name__ == "__main__":
    main()
