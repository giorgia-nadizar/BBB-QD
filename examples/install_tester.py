import jax.numpy as jnp
import gym
import evogym.envs
from evogym import sample_robot

if __name__ == '__main__':

    # test that jax and its numpy module is installed
    arr = jnp.asarray([1, 2, 3])
    print(arr)

    # test that evogym is installed
    body, _ = sample_robot((5, 5))
    env = gym.make('Walker-v0', body=body)
    env.reset()

    for _ in range(10):
        action = env.action_space.sample() - 1
        ob, reward, done, info = env.step(action)
        print(ob)

        if done:
            env.reset()

    env.close()
