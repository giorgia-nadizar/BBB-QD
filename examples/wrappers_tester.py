import gym
import evogym.envs
import numpy as np
from evogym import sample_robot, get_full_connectivity

from bbbqd.wrappers.action_wrappers import ActionSpaceCorrectionWrapper
from bbbqd.wrappers.observation_wrappers import GlobalObservationWrapper

if __name__ == '__main__':

    structure = np.array([[0, 0, 0], [3, 3, 0], [0, 0, 0]])
    env = gym.make('Walker-v0', body=structure, connections=get_full_connectivity(structure))
    flags = {
        'observe_voxel_vel': True,
        'observe_voxel_volume': True,
        # 'observe_time': True
    }
    env = ActionSpaceCorrectionWrapper(env)
    env = GlobalObservationWrapper(env, **flags)
    print(type(env))
    env.reset()


    for _ in range(10):
        action = env.action_space.sample() - 1
        ob, reward, done, info = env.step(action)
        print(ob)
        print(type(ob))
    #     env.render()
    #
    #     if done:
    #         env.reset()
    #
    # env.close()
