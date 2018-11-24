# Just importing this module automatically registers the environments
# noinspection PyUnresolvedReferences
import gym_minigrid
import gym_minigrid.wrappers as w
import gym

from gym import Env
from gym.envs.registration import EnvSpec
from vel.openai.baselines.bench import Monitor
from vel.rl.api.base import EnvFactory
from vel.rl.env.wrappers.clip_episode_length import ClipEpisodeLengthWrapper


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

        # Clipp episode length
        instance = ClipEpisodeLengthWrapper(instance, max_episode_length=100)

        # Monitor env
        instance = Monitor(instance, filename=None, allow_early_resets=False)

        return instance


def create(envname) -> MinigridEnvFactory:
    """ Vel creation function """
    return MinigridEnvFactory(envname)
