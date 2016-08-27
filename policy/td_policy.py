# -*- coding: utf-8 -*-
from policy import Policy  # lint:ok
from numpy import average


class TDPolicy(Policy):

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

    def estimateDelta(self, value, reward, gamma):
        vals = self.Q.getPossibleActions(self.nextState)
        return reward + gamma * average(vals) - value
