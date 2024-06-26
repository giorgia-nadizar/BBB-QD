from functools import partial
from pathlib import Path

import cv2
import numpy as np
import yaml
import jax.numpy as jnp
from moviepy.video.compositing.CompositeVideoClip import clips_array
from moviepy.video.io.VideoFileClip import VideoFileClip

from bbbqd.body.bodies import encode_body_directly
from bbbqd.body.body_utils import compute_body_encoding_function, compute_body_float_genome_length, compute_body_mask
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.wrappers import make_env
from qdax.core.gp.encoding import compute_encoding_function


def make_video(folder: str, render: bool = True, video_file_name: str = None, extra_prefix: str = "",
               position: int = None) -> None:
    config = yaml.safe_load(Path(f"{folder}/config.yaml").read_text())

    # Load fitnesses and genotypes
    try:
        fitnesses = jnp.load(f"{folder}/{extra_prefix}scores.npy")
    except FileNotFoundError:
        fitnesses = jnp.load(f"{folder}/{extra_prefix}fitnesses.npy")
    genotypes = jnp.load(f"{folder}/{extra_prefix}genotypes.npy")

    # Find best
    genome = genotypes[jnp.argmax(fitnesses)] if position is None else genotypes[position]

    # Define encoding function
    program_encoding_fn = compute_encoding_function(config)

    # Define function to return the proper controller
    controller_creation_fn = compute_controller_generation_fn(config)

    # Make body and controller
    if config.get("fixed_body", False):
        controller_genome = genome
        body = np.array(config["body"])
    else:
        body_float_length = compute_body_float_genome_length(config)
        body_mask = compute_body_mask(config)
        body_genome, controller_genome = jnp.split(genome, [len(body_mask) + body_float_length])
        body_encoding_fn = compute_body_encoding_function(config)
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

    # ga videos
    # body_names = ["biped-5x4", "worm-5x2", "tripod-5x5", "block-5x5", "evo-body-5x5"]
    # controller_names = ["global", "local"]
    # for body_name in body_names:
    #     for controller_name in controller_names:
    #         results_path = f"../results/{body_name}_{controller_name}_{seed}"
    #         video_file_path = f"../videos/{body_name}_{controller_name}_{seed}.avi"
    #         print(f"{body_name}, {controller_name}")
    #         make_video(results_path, render=False, video_file_name=video_file_path)

    # # me videos
    # base_info = "evo-body-5x5-me"
    # for sampler in ["all", "s1", "s2", "s3"]:
    #     extra_prefix = f"r1_"
    #     results_path = f"../results/{base_info}-{sampler}_{seed}"
    #     video_file_path = f"../videos/{base_info}-{sampler}_{seed}.avi"
    #     print(f"{sampler}")
    #     make_video(results_path, render=False, video_file_name=video_file_path, extra_prefix=extra_prefix)
    #
    # # collage of me videos
    # video_files = []
    # for sampler in ["all", "s1", "s2", "s3"]:
    #     video_path = f"../videos/evo-body-5x5-me-{sampler}_{seed}.avi"
    #     video_files.append(video_path)
    # video_clips = [VideoFileClip(file) for file in video_files]
    # collage = clips_array([video_clips])
    # collage.write_videofile(f"../videos/evo-body-5x5-me_{seed}.mp4", fps=24, codec='mpeg4')

    # videos for behavior analysis
    task = "climber"
    repertoire_prefix = "r3_"
    if task == "climber":
        repertoire_prefix = "extra_" + repertoire_prefix
    base_info = f"evo-body-10x10-{task}"
    # best video
    results_path = f"../results/me/{base_info}-all_{seed}"
    video_file_path = f"../videos/{task}_best_{seed}.avi"
    make_video(results_path, render=False, video_file_name=video_file_path, extra_prefix=repertoire_prefix, )
    # centroids video
    centroids_names = {
        413: "low-x+high-y",
        5: "high-x+high-y",
        866: "high-x+low-y",
        981: "low-x+low-y",
    }
    for centroid, name in centroids_names.items():
        print(centroid)
        print(name)
        results_path = f"../results/me/{base_info}-all_{seed}"
        video_file_path = f"../videos/{task}_behavior_{name}_{seed}.avi"
        make_video(results_path, render=False, video_file_name=video_file_path, extra_prefix=repertoire_prefix,
                   position=centroid)
    video_clips = [[VideoFileClip(f"../videos/{task}_behavior_low-x+high-y_{seed}.avi"),
                    VideoFileClip(f"../videos/{task}_behavior_high-x+high-y_{seed}.avi")],
                   [VideoFileClip(f"../videos/{task}_behavior_low-x+low-y_{seed}.avi"),
                    VideoFileClip(f"../videos/{task}_behavior_high-x+low-y_{seed}.avi")]]
    collage = clips_array(video_clips)
    collage.write_videofile(f"../videos/{task}_behavior_{seed}.mp4", fps=24, codec='mpeg4')
