import time
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros, Contra
from gym_super_mario_bros.actions import RIGHT_ONLY
from Contra.actions import RIGHT_ONLY
#from replay_agent_random_changed import DQNAgent
from mario_agent import DQNAgent
from wrappers import wrapper

# Build env (first level, right only)
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, RIGHT_ONLY)
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


agent.replay(env, "/Users/karanmaheshwari/PycharmProjects/Mario/models_copy/", 10000, False)
