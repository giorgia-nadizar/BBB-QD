from functools import partial
from pathlib import Path
from typing import Dict

import cv2
import gym
import evogym.envs
import numpy as np
import yaml
from jax import random
import jax.numpy as jnp

from bbbqd.body.bodies import encode_body
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller_and_body
from bbbqd.wrappers import make_env
from qdax.core.gp.encoding import compute_encoding_function
from qdax.core.gp.utils import update_config


def make_video(folder: str, render: bool = True, video_file_name: str = None):
    config = yaml.safe_load(Path(f"{folder}/config.yaml").read_text())

    # Load fitnesses and genotypes
    fitnesses = jnp.load(f"{folder}/scores.npy")
    genotypes = jnp.load(f"{folder}/genotypes.npy")

    # Find best
    genome = genotypes[jnp.argmax(fitnesses)].astype(int)

    # Define encoding function
    program_encoding_fn = compute_encoding_function(config)

    # Define function to return the proper controller
    controller_creation_fn = compute_controller_generation_fn(config)

    # Make body and controller
    if config.get("fixed_body", False):
        controller_genome = genome
        body = np.array(config["body"])
    else:
        body_genome, controller_genome = jnp.split(genome, [config["grid_size"] ** 2])
        body_encoding_fn = partial(encode_body, make_connected=True)
        body = body_encoding_fn(body_genome)

    controller = controller_creation_fn(program_encoding_fn(controller_genome))

    # Make environment
    env = make_env(config, body)
    cumulative_reward = 0
    obs = env.reset()
    if video_file_name is not None:
        sample_img = env.render('img')
        size = sample_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_file_name, fourcc, 10, (size[1], size[0]))
    for _ in range(config["episode_length"]):
        action = controller.compute_action(obs)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        if render:
            env.render()
        if video_file_name is not None:
            img = env.render('img')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img)
        if done:
            break

    env.close()

    if video_file_name is not None:
        cv2.destroyAllWindows()
        video.release()


if __name__ == '__main__':
    seed = 0
    body_names = ["biped-5x4", "worm-5x2", "tripod-5x5", "block-5x5", "evo-body-5x5"]
    controller_names = ["global", "local"]

    for body_name in body_names:
        for controller_name in controller_names:
            results_path = f"../results/{body_name}_{controller_name}_{seed}"
            video_file_path = f"../videos/{body_name}_{controller_name}_{seed}.avi"
            print(f"{body_name}, {controller_name}")
            make_video(results_path, render=False, video_file_name=video_file_path)