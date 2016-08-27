# -*- coding: utf-8 -*-
from state_action import StateAction
from policy import Policy


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

    def updateEpisode(self):
        states, actions, rewards = self.history.getSequence()
        states_evaluated = []
        for t, state in enumerate(states):
            representations = self.Q.getRepresentation(
                                            StateAction(state, actions[t]))
            ret = self.history.getReturn(t)
            for r in representations:
                if r not in states_evaluated:
                    self.Q.addDeltaToQValue(r, self.estimateDelta(
                                                self.Q.Q[r],
                                                self.updater.alfa, ret))
                    states_evaluated.append(r)

    def estimateDelta(self, value, alfa, vt):
        return alfa * (vt - value)
