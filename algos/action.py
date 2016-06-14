# -*- coding: utf-8 -*-


class Action:

    def __init__(self, id):
        self.id = id
        self.value = 1
        self.visits = 0

    def updateValue(self, updater):
        self.value = updater.estimateActionValue()

    def addVisit(self):
        self.visits += 1

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(frozenset(self.id))

    def __str__(self):
        return "Action[id: " + str(self.id) + ", value: " + \
                                            str(self.value) + "]"
