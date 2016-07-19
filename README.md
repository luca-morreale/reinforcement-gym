# reinforcement-gym
Set of classes in python to play with the gym of openai

# Prerequisites
You have to install openai-gym, check [here](https://gym.openai.com/docs) how to do it.

**Watch out**, this code has been tested only with Python 3.5

# How to run
The main_parser make relatively simple run any kind of environment and agent.

First of all choose the *enviroment*:
* CartPole-v0
* MountainCar-v0 

There is no reason why you could not test other enviroments but for the moment I still working with this two.
To run the others you have to create your own main file.

You have to decide the *policy* the agent will use:
* sarsa
* td (this implemention perform an average on all the action-value of each possible future action)
* mc
* q

The type of *generalization* on the state you want to apply:
* none
* norm
* hash
* tiles

The type of the *updater*:
* normal (update only the current state)
* trace (replacing traces)

The learning rate parameter *alfa*, the discount factor *gamma*.

Then it is possible to set some optiona paramters, some of 
them depends also on the settings you choose.

You can set:
- *lambda* known as trace decay parameter
- *epsilon* in case you would use an epsilon-greedy chooser
- the kind of *action chooser* 
  - greedy for choose always the optimal action
  - epsilon for choose the epslion-greedy best action
- the size of the *cell* in case you choose a norm or hash generalization
- *tilings* how many of tiling to use in case a tile coding has been chosen
- *tiles* number of tiles for each tiling
- *n* the maximun number of possible states after the discretization
- *limit* after how many episodes before switch to optimal policy, in case QLearning has been chosen
- *episodes* the maximum number of episodes to run
- *record* if record the experiment
- *directory* where to save the recording

Finally to run everything you just have to encode your setting and launch it, for instance:
```python
python main_parser.py -env=CartPole-v0 -agent=q -generalization=tiles -updater=trace -alfa=0.5 -gamma=1 -limit=70
```

Do not worry if you forget some parameters, many of them have defaults values and are used only if required.
