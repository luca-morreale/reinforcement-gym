# -*- coding: utf-8 -*-
from parser.parser import ArgsParser


def main():
    parser = ArgsParser()
    env = parser.getEnv()

    pi = parser.getAgent()

    if parser.getRecordFlag():
        env.monitor.start('./' + parser.getDirectoryName())

    n_episodes = parser.getEpisodesNumber()
    for episode in range(1, n_episodes):
        pi.doEpisode(episode)

    if parser.getRecordFlag():
        env.monitor.close()


if __name__ == "__main__":
    main()