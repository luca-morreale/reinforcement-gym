# -*- coding: utf-8 -*-
from scipy import *

from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q
#from pybrain.rl.learners import SARSA

import gym


def doEpisode(env, agent, episode_n):
    agent.newEpisode()
    state = env.reset()
    step = 0
    while True:
        #env.render()
        step += 1
        print(state)
        agent.integrateObservation(state)

        action = agent.getAction()
        print(action)
        observation, reward, done, info = self.env.step(action)

        agent.giveReward(reward)

        if done or step > env.spec.timestep_limit:
            print(('finished episode', episode_n, 'steps', step))
            break


def main():
    env = gym.make('CartPole-v0')

    controller = ActionValueTable(1, 2)
    controller.initialize(1.)

    learner = Q()
    agent = LearningAgent(controller, learner)

    for episode in range(1, 15000):
        doEpisode(env, agent, episode)


if __name__ == "__main__":
    main()