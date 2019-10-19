from keras.callbacks import TensorBoard
from rl.memory import SequentialMemory
import numpy as np
from time import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from contra.CustomContra.actions import CUSTOM_MOVEMENT
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from nes_py.wrappers import JoypadSpace
import gym
import argparse
import json
import matplotlib.pyplot as plt

# def visualize_log(filename, figsize=None, output=None):
#     with open(filename, 'r') as f:
#         data = json.load(f)
#     if 'episode' not in data:
#         raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
#     episodes = data['episode']
#
#     # Get value keys. The x axis is shared and is the number of episodes.
#     keys = sorted(list(set(data.keys()).difference(set(['episode']))))
#
#     if figsize is None:
#         figsize = (15., 5. * len(keys))
#     f, axarr = plt.subplots(len(keys), sharex=True, figsize=figsize)
#     for idx, key in enumerate(keys):
#         axarr[idx].plot(episodes, data[key])
#         axarr[idx].set_ylabel(key)
#     plt.xlabel('episodes')
#     plt.tight_layout()
#     if output is None:
#         plt.show()
#     else:
#         plt.savefig(output)

ENV_NAME = 'CustomContra-v2'
# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env = JoypadSpace(env, CUSTOM_MOVEMENT)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
print(nb_actions)
print(env.observation_space.shape)
obs_dim = env.observation_space.shape[0]


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
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg')
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

weights_filename = '../logs/dqn_contra_weights.h5f'
# checkpoint_weights_filename = 'dqn_contra_weights_step.h5f'
# log_filename = 'dqn_contra_log.json'

tensorboard = TensorBoard(log_dir="../tensorboard-logs/{}".format(time()))

# callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250)]
# callbacks += [FileLogger(log_filename, interval=100)]

dqn.fit(env, callbacks=[tensorboard], nb_steps=5000, log_interval=100, visualize=True, verbose=1)


# After training is done, we save the final weights.
dqn.save_weights(weights_filename, overwrite=True)

# You can use visualize_log to easily view the stats that were recorded during training. Simply
# provide the filename of the `FileLogger` that was used in `FileLogger`.
# visualize_log(log_filename)


# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)