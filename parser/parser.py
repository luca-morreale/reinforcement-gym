# -*- coding: utf-8 -*-
import argparse
import gym

from policy.mc_policy import MCPolicy
from policy.td_policy import TDPolicy
from policy.sarsa_policy import SarsaPolicy
from policy.q_learning_policy import QLearningPolicy

from action_chooser.epsilon_greedy_chooser import EpsilonGreedyChooser

from generalizer.none_state_generalizer import NoneGeneralizer
from generalizer.norm_state_generalizer import NormGeneralizer
from generalizer.hash_state_generalizer import HashGeneralizer
from generalizer.tiles_state_generalizer import TilesStateGeneralizer

from updater.updater import Updater
from updater.trace_updater import UpdaterTraced


class Args():
    pass


class ArgsParser():

    def __init__(self):
        parser = argparse.ArgumentParser(description='Solve RL problems')
        parser.add_argument('-env', default='CartPole-v0', type=str,
                            choices=['CartPole-v0', 'MountainCar-v0'],
                            required=True,
                            help='name of the enviroment')
        parser.add_argument('-agent', default='sarsa', type=str,
                            choices=['sarsa', 'td', 'mc', 'q'],
                            required=True,
                            help='type of the agent to use')
        parser.add_argument('-generalization', default='tile', type=str,
                            choices=['none', 'norm', 'hash', 'tiles'],
                            required=True,
                            help='type of the generalization ' +
                            'the agent will use')

        parser.add_argument('-updater', default='normal', type=str,
                            required=True, choices=['normal', 'trace'],
                            help='type of updater for Q')

        parser.add_argument('-alfa', default=0.5, type=float, required=True,
                            help='value of the learning rate')

        parser.add_argument('-gamma', default=1, type=float, required=True,
                            help='value of the discount factor')

        # not required if used a QLearning agent
        parser.add_argument('-epsilon', default=0.1, type=float,
                            help='probability of make a random choice')

        # not required if used a normal updater
        parser.add_argument('-lambda_', default=0.9, type=float,
                            help='elegibility trace decay factor')

        # not required if the agent is a QLearning
        parser.add_argument('-action', default='epsilon', type=str,
                            choices=['greedy', 'epsilon'],
                            help='type of action chooser')

        # required only if tile generalizer has been chosen
        parser.add_argument('-tiles', default=5, type=int,
                            help='number of tiles for each tiling')

        # required only if tile generalizer has been chosen
        parser.add_argument('-tilings', default=2, type=int,
                            help='number of tilings')

        # required only if tile generalizer has been chosen
        parser.add_argument('-n', default=30000, type=int,
                            help='number of different tiles')

        # required only if norm or hash generalizer has been chosen
        parser.add_argument('-cell', default=0.5, type=float,
                            help='size of a cell')

        # required only if QLearning has been selected
        parser.add_argument('-limit', default=100, type=int,
                            help='episodes to wait before apply greedy policy')

        parser.add_argument('-record', default=False, type=bool,
                            help='record the execution')

        parser.add_argument('-episodes', default=2000, type=int,
                            help='number of episodes to execute')

        parser.add_argument('-directory', default='experiment', type=str,
                            help='name of the directory where save' +
                            ' the experiment')

        self.args = Args()
        parser.parse_args(namespace=self.args)

        self.env = None
        self.agent = None
        self.action = None
        self.updater = None
        self.generalizer = None

    def getEnv(self):
        if self.env is None:
            self.env = gym.make(self.args.env)
        return self.env

    def getM(self):
        return self.getEnv().action_space.n

    def getActionChooser(self):
        if self.action is None:
            self.action = self._getActionChooser()
        return self.action

    def _getActionChooser(self):
        if self.args.agent == 'q':
            return None
        elif self.args.action == 'greedy':
            return EpsilonGreedyChooser(1, self.getEnv().action_space.n)
        else:
            return EpsilonGreedyChooser(self.getEpsilon(),
                                    self.getEnv().action_space.n)

    def getEpsilon(self):
        e = self.args.epsilon
        if e > 1:
            e = e % 1
        return e

    def getGeneralizer(self):
        if self.generalizer is None:
            self.generalizer = self._getGeneralizer()
        return self.generalizer

    def _getGeneralizer(self):
        if self.args.generalization == 'none':
            return NoneGeneralizer(self.getM())
        elif self.args.generalization == 'norm':
            return NormGeneralizer(self.getM(), self.args.cell)
        elif self.args.generalization == 'hash':
            return HashGeneralizer(self.getM(), self.args.cell,
                                                self.getEnv().observation_space)
        else:
            return TilesStateGeneralizer(self.args.tilings, self.args.tiles,
                                            self.getEnv().observation_space,
                                            self.getM(), self.args.n)

    def getUpdater(self):
        if self.updater is None:
            self.updater = self._getUpdater()
        return self.updater

    def _getUpdater(self):
        generalizer = self.getGeneralizer()
        if self.args.updater == 'normal':
            return Updater(self.args.gamma, self.args.alfa, generalizer)
        else:
            return UpdaterTraced(self.args.gamma, self.args.alfa,
                            generalizer, self.args.lambda_)

    def getAgent(self):
        action_chooser = self.getActionChooser()
        generalizer = self.getGeneralizer()
        updater = self.getUpdater()

        if self.args.agent == 'mc':
            pi = MCPolicy(action_chooser, generalizer, updater)
        elif self.args.agent == 'td':
            pi = TDPolicy(action_chooser, generalizer, updater)
        elif self.args.agent == 'sarsa':
            pi = SarsaPolicy(action_chooser, generalizer, updater)
        else:
            pi = QLearningPolicy(generalizer, updater, self.getM(),
                                                            self.args.limit)
        pi.set(self.getEnv())
        return pi

    def getRecordFlag(self):
        return self.args.record

    def getEpisodesNumber(self):
        return self.args.episodes

    def getDirectoryName(self):
        return self.args.directory