# -*- coding: utf-8 -*-
from policy import Policy  # lint:ok
from state import State  # lint:ok


class TDPolicy(Policy):

    #
    def doEpisode(self, episode_n):
        observation = self.env.reset()
        step = 0
        while True:
            last_state = State(observation)
            #env.render()
            step += 1

            act = self.getAction(last_state)
            observation, reward, done, info = self.env.step(act.id)

            self.history.addStep(last_state, act, reward)
            self.updateStep(last_state, act, reward, step-1)

            if done or step > self.env.spec.timestep_limit:
                print(('finished episode', episode_n, 'steps', step))
                break

        self.epsilon = 1 / episode_n

    def estimateNewValue(self, value, alfa, vt, t):
        vt = self.history.rewards[t]
        return value + alfa * (vt + self.gamma * value - value)