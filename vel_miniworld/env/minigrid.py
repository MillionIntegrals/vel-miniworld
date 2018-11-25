# Just importing this module automatically registers the environments
# noinspection PyUnresolvedReferences
import gym_minigrid
import gym_minigrid.wrappers as w
import gym
import math

from gym import Env
from gym.envs.registration import EnvSpec
from vel.openai.baselines.bench import Monitor
from vel.rl.api.base import EnvFactory
from vel.rl.env.wrappers.clip_episode_length import ClipEpisodeLengthWrapper


class StateBonus(gym.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.counts = {}
        self.scale = scale

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = tuple(env.agent_pos.tolist())

        # Get the count for this key
        preCnt = 0
        if tup in self.counts:
            preCnt = self.counts[tup]

        # Update the count for this key
        newCnt = preCnt + 1
        self.counts[tup] = newCnt

        bonus = 1 / math.sqrt(newCnt)

        reward += bonus * self.scale

        return obs, reward, done, info


class MinigridEnvFactory(EnvFactory):
    def __init__(self, envname):
        self.envname = envname

    def specification(self) -> EnvSpec:
        """ Return environment specification """
        return gym.spec(self.envname)

    def instantiate(self, seed=0, serial_id=1, preset='default', extra_args=None) -> Env:
        """ Create a new Env instance """
        instance = gym.make(self.envname)
        instance.seed(seed + serial_id)

        # Flat observation wrapper
        instance = w.ImgObsWrapper(instance)

        # State exploration bonus?
        # instance = StateBonus(instance)

        # Clipp episode length
        instance = ClipEpisodeLengthWrapper(instance, max_episode_length=100)

        # Monitor env
        instance = Monitor(instance, filename=None, allow_early_resets=False)

        return instance


def create(envname) -> MinigridEnvFactory:
    """ Vel creation function """
    return MinigridEnvFactory(envname)
