# -*- coding: utf-8 -*-


class StateAction():

    def __init__(self, _id, value=1):
        self.id = _id
        self.value = value
        self.visits = 0

    def addVisits(self):
        self.visits += 1

    def __str__(self):
        return "StateAction[id: " + str(self.id) + \
                    " , value: " + str(self.value) + "]"