import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import EpisodeParameterMemory
from nes_py.wrappers import JoypadSpace
import gym
from rl.core import Processor
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from PIL import Image

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

ENV_NAME = 'Contra-v0'
# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]
print(nb_actions, obs_dim)
# Option 1 : Simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

# Option 2: deep network
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('softmax'))


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

policy = BoltzmannQPolicy()
processor = AtariProcessor()

print(model.summary())
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = EpisodeParameterMemory(limit=1000, window_length=1)
cem = DQNAgent(model=model, policy=policy, nb_actions=nb_actions, memory=memory,
			   processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
			   train_interval=4, delta_clip=1.)
'''nb_actions=nb_actions, memory=memory,
batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)'''
cem.compile(optimizer=Adam(lr=.00025))
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
cem.fit(env, nb_steps=1000, visualize=True, verbose=2)
# After training is done, we save the best weights.
cem.save_weights('cem_{}_params.h5f'.format(ENV_NAME), overwrite=True)
# Finally, evaluate our algorithm for 5 episodes.
cem.test(env, nb_episodes=5, visualize=True)
