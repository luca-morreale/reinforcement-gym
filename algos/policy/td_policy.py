# -*- coding: utf-8 -*-
from policy.policy import Policy  # lint:ok
from state import State  # lint:ok


class TDPolicy(Policy):

    #
    def doEpisode(self, episode_n):
        observation = self.env.reset()
        step = 0
        while True:
            last_state = State(observation, self.cellSize)
            if self.show:
                self.env.render()
            step += 1

            act = self.getAction(last_state)
            observation, reward, done, info = self.env.step(act.id)

            self.appendToHistory(last_state, act, reward)
            self.update(last_state, act, reward, step - 1)

            if done or step > self.env.spec.timestep_limit:
                print(('finished episode', episode_n, 'steps', step))
                break
        self.newEpisode()

    def estimateDelta(self, value, alfa, gamma, vt, rt):
        return alfa[0] * (rt + gamma * value - value)