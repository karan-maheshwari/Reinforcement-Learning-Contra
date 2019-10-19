from rl.memory import SequentialMemory
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from contra.CustomContra.actions import CUSTOM_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym


ENV_NAME = 'CustomContra-v2'
# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env = JoypadSpace(env, CUSTOM_MOVEMENT)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=5000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae','acc'])

weights_filename = '../logs/dqn_contra_weights.h5f'

dqn.load_weights(weights_filename)
# training_history = dqn.fit(env, callbacks=[tensorboard], nb_steps=0, log_interval=100, visualize=True, verbose=1, nb_max_episode_steps=1000)


dqn.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=1000)