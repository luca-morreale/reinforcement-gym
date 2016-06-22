# -*- coding: utf-8 -*-
from policy.policy import Policy  # lint:ok
from state import State  # lint:ok


class SarsaPolicy(Policy):

    #
    def doEpisode(self, episode_n):
        observation = self.env.reset()
        self.nextState = State(observation, self.cellSize)
        self.nextAction = self.getAction(self.nextState)

        step = 0
        while True:
            last_state = self.nextState
            act = self.nextAction
            if self.show:
                self.env.render()
            step += 1

            observation, reward, done, info = self.env.step(act.id)

            self.appendToHistory(last_state, act, reward)
            self.nextState = State(observation, self.cellSize)

            self.updateStep(last_state, act, reward, step - 1)
            #self.prettyPrintQ()
            if done or step > self.env.spec.timestep_limit:
                print(('finished episode', episode_n, 'steps', step))
                break
        self.newEpisode()

    def estimateDelta(self, value, alfa, gamma, vt, rt):
        self.nextAction = self.getAction(self.nextState)
        return alfa[0] * (rt + gamma *
                                self.nextAction.value - value)
