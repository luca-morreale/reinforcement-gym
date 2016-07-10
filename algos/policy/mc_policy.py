# -*- coding: utf-8 -*-
from policy.policy import Policy


class MCPolicy(Policy):

    def doEpisode(self, episode_n):

        observation = self.env.reset()
        step = 0

        while True:
            last_state = observation
            action = self.getAction(last_state)
            step += 1

            if self.show:
                self.env.render()

            observation, reward, done, info = self.env.step(action)

            self.appendToHistory(last_state, action, reward)

            if done or step > self.env.spec.timestep_limit:
                print(('finished episode', episode_n, 'steps', step))
                break

        self.updateEpisode()

    def estimateDelta(self, vt):
        pass
