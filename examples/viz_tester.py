import gym
import evogym.envs
from evogym import sample_robot

if __name__ == '__main__':

    body, connections = sample_robot((5, 5))
    env = gym.make('Carrier-v0', body=body)
    env.reset()

    for _ in range(1000):
        print(env.action_space)

        action = env.action_space.sample() - 1
        ob, reward, done, info = env.step(action)
        env.render()

        if done:
            env.reset()

    env.close()
