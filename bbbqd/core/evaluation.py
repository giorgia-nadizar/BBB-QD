from typing import Union, Dict, Any

from bbbqd.brain.controllers import ControllerWrapper, Controller
from bbbqd.wrappers import make_env


# TODO: will take care of body variability
# TODO: will also provide descriptors
def evaluate_controller(controller: Union[Controller, ControllerWrapper], config: Dict[str, Any]) -> float:
    env = make_env(config)

    cumulative_reward = 0
    obs = env.reset()
    for _ in range(config["episode_length"]):
        action = controller.compute_action(obs)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
    return cumulative_reward
