# -*- coding: utf-8 -*-
from copy import copy


class StateAction():

    def __init__(self, obs, action, value=1):
        self.obs = obs
        self.action = copy(action)
        self.value = value

    def __str__(self):
        return "StateAction[obs: " + str(self.obs) + \
                    "action: " + str(self.id) + \
                    " , value: " + str(self.value) + "]"