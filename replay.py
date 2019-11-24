import time
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros, Contra
from gym_super_mario_bros.actions import RIGHT_ONLY
#from Contra.actions import RIGHT_ONLY
#from replay_agent_random_changed import DQNAgent
from replay_agent import DQNAgent
from wrappers import wrapper

CUSTOM = [
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['NOOP']
]

CUSTOM_MOVEMENT = [
    ['NOOP'],
   ['right'],
   ['right', 'A'],
   ['right', 'B'],
   ['right', 'A', 'B'],
   ['right', 'B', 'up'],
   ['right', 'A', 'B', 'up'],
   ['B'],
   ['A', 'B'],
   ['down', 'A'],
   ['down', 'B'],
   ['down', 'A', 'B'],
   ['up', 'A'],
   ['up', 'A', 'B']
]


# Build env (first level, right only)
env = Contra.make('Contra-v0')
env = JoypadSpace(env, CUSTOM)
env = wrapper(env)

# Parameters
states = (84, 84, 4)
actions = env.action_space.n
# Parameters

# Agent
# Agent
agent = DQNAgent(states=states, actions=actions, max_memory=100000, double_q=True)

# Episodes
episodes = 10000
rewards = []

# Timing
start = time.time()
step = 0
# Main loop


agent.replay(env, "/Users/karanmaheshwari/PycharmProjects/Mario/models/", 10000, False)
