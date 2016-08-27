# -*- coding: utf-8 -*-
from policy import Policy
from action_chooser.epsilon_greedy_chooser import EpsilonGreedyChooser
from numpy import max


class QLearningPolicy(Policy):

    def __init__(self, Q, updater, m, randomEpisodes):
        self.randomEpisodes = randomEpisodes
        self.actionChooser = EpsilonGreedyChooser(1, m)
        self.Q = Q
        self.updater = updater
        self.newEpisode()

    def doEpisode(self, episode_n):
        observation = self.env.reset()

        self.nextState = observation
        step = 0

        while True:
            last_state = self.nextState
            action = self.getAction(last_state)
            step += 1

            if self.show:
                self.env.render()

            observation, reward, done, info = self.env.step(action)

            self.appendToHistory(last_state, action, reward)
            self.nextState = observation

            #self.Q.prettyPrintQ()
            if done or step > self.env.spec.timestep_limit:
                print(('finished episode', episode_n, 'steps', step))
                break
            self.update(last_state, action, reward)
        self.newEpisode()

    def newEpisode(self):
        Policy.newEpisode(self)
        if self.randomEpisodes > 0:
            self.randomEpisodes -= 1
            self.actionChooser.epsilon = 1
        else:
            self.actionChooser.epsilon = 0

    def estimateDelta(self, value, reward, gamma):
        maxQ = max(self.Q.getPossibleActions(self.nextState))
        return reward + gamma * maxQ - value
