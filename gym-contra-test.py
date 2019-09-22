from nes_py.wrappers import JoypadSpace
import gym
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

env = gym.make('Contra-v0')

env = JoypadSpace(env, RIGHT_ONLY)

print("actions", env.action_space)
print("observation_space ", env.observation_space.shape[0])
f = open("test.txt", "w+")
done = False
env.reset()
for step in range(1000):

	if done:
		print("Over")
		break
	state, reward, done, info = env.step(env.action_space.sample())
	print(reward, done, info)
	print()
	env.render()
	state.tofile('test_num.txt'.replace('num', str(step)), format='str')
	f.write("\n")

env.close()
