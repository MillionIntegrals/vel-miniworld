# vel-miniworld
An example project using vel to train reinforcement learning agents
on existing community gym environments. A work in progress repository.

Currently supported environments:
- https://github.com/maximecb/gym-minigrid

Work in progress:
- https://github.com/maximecb/gym-miniworld

Examples confirmed to be working currently:
- `examples-configs/ppo/ppo_minigrid_empty_8x8.yaml`
- `examples-configs/ppo/ppo_minigrid_doorkey_6x6.yaml` (doesn't converge every time)


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


## Some animations

<p align="center">
<img src="/animations/ppo_minigrid_empty_8x8.gif">
</p>

<p align="center">
<img src="/animations/ppo_minigrid_doorkey_6x6.gif">
</p>
