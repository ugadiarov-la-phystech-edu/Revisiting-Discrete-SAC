import gym
import numpy as np

import envs
from tianshou.env import ShmemVectorEnv


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize observations to 0~1.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        low = np.min(env.observation_space.low)
        high = np.max(env.observation_space.high)
        self.bias = low
        self.scale = high - low
        self.observation_space = gym.spaces.Box(
            low=0., high=1., shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        return (observation - self.bias) / self.scale


class ChannelsFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = (env.observation_space.shape[2], *env.observation_space.shape[:2])
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape,
            dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


def wrap_shapes2d(env_id):
    env = gym.make(env_id)
    env = ChannelsFirst(env)
    env = ScaledFloatFrame(env)

    return env


def make_env(task, seed, training_num, test_num):
    env = wrap_shapes2d(task)
    train_envs = ShmemVectorEnv(
        [
            lambda:
            wrap_shapes2d(task)
            for _ in range(training_num)
        ]
    )
    test_envs = ShmemVectorEnv(
        [
            lambda:
            wrap_shapes2d(task)
            for _ in range(test_num)
        ]
    )
    env.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
    return env, train_envs, test_envs