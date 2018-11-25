# vel-miniworld
An example project using vel to train reinforcement learning agents
on existing community gym environments. A work-in-progress repository.


Supported environments:
- https://github.com/maximecb/gym-minigrid
- https://github.com/maximecb/gym-miniworld


Examples confirmed to be working:
- `examples-configs/ppo/ppo_minigrid_empty_8x8.yaml`
- `examples-configs/ppo/ppo_minigrid_doorkey_6x6.yaml` (doesn't converge every time)
- `examples-configs/ppo/ppo_miniworld_hallway.yaml`


## How to run?

```bash
git clone git@github.com:MillionIntegrals/vel-miniworld.git
cd vel-miniworld

# Optionally, if you don't want to store metrics in the db and visualize in VisDom
mv .velproject.dummy.yaml .velproject.yaml

pipenv install
pipenv shell
vel examples-configs/ppo/ppo_minigrid_empty_8x8.yaml train
vel examples-configs/ppo/ppo_minigrid_empty_8x8.yaml record

# Optionally, play a video of agent solving a rather simple environment
mplayer output/videos/ppo_minigrid_empty_8x8/0/ppo_minigrid_empty_8x8_vid_0010.avi
```

## Additional notes

For the textures to load properly for the 3D rendered `miniworld` environment, it needs to be installed
from a git repository, by running `pip install -e .` in the top-level directory of the checkout.

Let me know if you have any other problems running the environments.


## Some animations

Solving simple small gridworld environment:

<p align="center">
<img src="/animations/ppo_minigrid_empty_8x8.gif">
</p>

Solving slightly more complex gridworld environment with sparse rewards:

<p align="center">
<img src="/animations/ppo_minigrid_doorkey_6x6.gif">
</p>

Solving small 3D rendered world:

<p align="center">
<img src="/animations/ppo_miniworld_hallway.gif">
</p>
