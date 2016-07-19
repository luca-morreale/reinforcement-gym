# -*- coding: utf-8 -*-
from policy.policy import Policy


class SarsaPolicy(Policy):

    def doEpisode(self, episode_n):
        observation = self.env.reset()

        self.setNextStateAction(observation)
        step = 0

        while True:
            last_state = self.nextState
            action = self.nextAction
            step += 1

            if self.show:
                self.env.render()

            observation, reward, done, info = self.env.step(action)

            self.appendToHistory(last_state, action, reward)
            self.setNextStateAction(observation)

            #self.Q.prettyPrintQ()
            if done or step > self.env.spec.timestep_limit:
                print(('finished episode', episode_n, 'steps', step))
                break
            self.update(last_state, action, reward)
        self.newEpisode()

    def setNextStateAction(self, observation):
        self.nextState = observation
        self.nextAction = self.getAction(self.nextState)

    def estimateDelta(self, value, reward, gamma):
        nextVal = self.Q.getCombinedValue(self.nextState, self.nextAction)
        return reward + gamma * nextVal - value
