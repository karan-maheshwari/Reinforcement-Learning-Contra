"""Registration code of Gym environments in this package."""
import gym


def _register_Contra_env(Name):

    gym.envs.registration.register(
        id=Name,
        entry_point='contra.CustomContra.env_contra:ContraEnv',
        max_episode_steps=9999999,
        reward_threshold=32000,
        kwargs={},
        nondeterministic=True
    )


_register_Contra_env("CustomContra-v2")

# create an alias to gym.make for ease of access
make = gym.make

# # define the outward facing API of this module (none, gym provides the API)
__all__ = [make.__name__]
