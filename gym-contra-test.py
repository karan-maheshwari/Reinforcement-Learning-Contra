from nes_py.wrappers import JoypadSpace
import gym
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

env = gym.make('Contra-v0')
env = JoypadSpace(env, RIGHT_ONLY)

print("actions", env.action_space)
print("observation_space ", env.observation_space.shape[0])

done = False
env.reset()
for step in range(1):
	'''
	print(self._life,)
	print(self._is_dead,)
	print(self._get_done(),)
	print(self._score(),)
	print(self._player_state,)
	print(self._x_position,)
	print(self._y_position,)
	'''
	if done:
		print("Over")
		break
	state, reward, done, info = env.step(env.action_space.sample())
	print(reward, done, info)
	env.render()

env.close()