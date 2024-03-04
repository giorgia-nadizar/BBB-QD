import gym
import evogym.envs
import numpy as np

from bbbqd.wrappers.controller_wrappers import GlobalWrapper, LocalWrapper

if __name__ == '__main__':

    structure = np.array([[0, 0, 0], [3, 3, 3], [0, 0, 0]])
    env = gym.make('Walker-v0', body=structure)

    flags = {
        'observe_voxel_vel': True,
        'observe_voxel_volume': True,
        'observe_time': True,
        'wrapper': 'global'
    }
    if flags.get('wrapper', None) == 'global':
        env = GlobalWrapper(env, **flags)
    elif flags['wrapper'] == 'local':
        env = LocalWrapper(env, **flags)

    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
    env.close()
