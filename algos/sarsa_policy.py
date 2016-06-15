# -*- coding: utf-8 -*-
from policy import Policy  # lint:ok
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
            #self.env.render()
            step += 1

            observation, reward, done, info = self.env.step(act.id)

            self.history.addStep(last_state, act, reward)
            self.nextState = State(observation, self.cellSize)

            self.updateStep(last_state, act, reward, step - 1)

            if done or step > self.env.spec.timestep_limit:
                print(('finished episode', episode_n, 'steps', step))
                break

        self.epsilon = 1 / episode_n

    def estimateNewValue(self, value, alfa, vt, t):
        vt = self.history.rewards[t]
        self.nextAction = self.getAction(self.nextState)
        return value + alfa[0] * (vt + self.gamma *
                                self.nextAction.value - value)
