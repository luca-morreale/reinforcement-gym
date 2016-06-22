# -*- coding: utf-8 -*-
from policy.policy import Policy
from state import State


class MCPolicy(Policy):

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

            self.appendToHistory(last_state, act, reward)

            if done or step > self.env.spec.timestep_limit:
                print(('finished episode', episode_n, 'steps', step))
                break

        self.updateEpisode()

    def estimateDelta(self, value, alfa, vt, rt):
        return alfa[1] * (vt - value)

