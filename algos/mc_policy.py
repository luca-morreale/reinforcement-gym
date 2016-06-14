# -*- coding: utf-8 -*-
from policy import Policy
from state import State


class MCPolicy(Policy):

    def estimateNewValue(self, value, alfa, vt, t):
        return value + alfa[1] * (vt - value)

    #
    def doEpisode(self, episode_n):

        observation = self.env.reset()
        step = 0
        while True:
            last_state = State(observation, self.cellSize)
            self.env.render()
            step += 1

            act = self.getAction(last_state)
            observation, reward, done, info = self.env.step(act.id)

            self.history.addStep(last_state, act, reward)

            if done or step > self.env.spec.timestep_limit:
                print(('finished episode', episode_n, 'steps', step))
                break

        self.updateEpisode()
        self.epsilon = 1 / episode_n

