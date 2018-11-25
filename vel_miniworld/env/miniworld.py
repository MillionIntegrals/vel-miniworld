# Just importing this module automatically registers the environments
# noinspection PyUnresolvedReferences
import gym_miniworld
import gym

from gym import Env
from gym.envs.registration import EnvSpec

from vel.openai.baselines.bench import Monitor
from vel.rl.api.base import EnvFactory
from vel.rl.env.wrappers.clip_episode_length import ClipEpisodeLengthWrapper


class MiniworldEnvFactory(EnvFactory):
    def __init__(self, envname):
        self.envname = envname

    def specification(self) -> EnvSpec:
        """ Return environment specification """
        return gym.spec(self.envname)

    def instantiate(self, seed=0, serial_id=1, preset='default', extra_args=None) -> Env:
        """ Create a new Env instance """
        instance = gym.make(self.envname)
        instance.seed(seed + serial_id)

        # Clip episode length, should be parametrized etc
        instance = ClipEpisodeLengthWrapper(instance, max_episode_length=1000)

        # Monitor env
        instance = Monitor(instance, filename=None, allow_early_resets=False)

        return instance


def create(envname) -> MiniworldEnvFactory:
    """ Vel creation function """
    return MiniworldEnvFactory(envname)
