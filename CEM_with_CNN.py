import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from nes_py.wrappers import JoypadSpace
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


ENV_NAME = 'Contra-v0'

CUSTOM_MOVEMENT = [
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
]


env = gym.make(ENV_NAME)
env = JoypadSpace(env, RIGHT_ONLY)
np.random.seed(120)
env.seed(120)
nb_actions = env.action_space.n

print(env.observation_space.shape)
print(env)


model = Sequential()
model.add(Flatten(input_shape=(1,)+env.observation_space.shape))
model.add(Reshape(env.observation_space.shape))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=env.observation_space.shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))
print(model.summary())


memory = SequentialMemory(limit=5000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


dqn.fit(env, nb_steps=100000, visualize=False)


dqn.save_weights('dqn_{}_weights.h5f'.format('test_run'), overwrite=True)


#dqn.test(env, nb_episodes=5, visualize=True)