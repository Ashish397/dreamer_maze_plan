# ============================================================================
#   IMPORTS
# ----------------------------------------------------------------------------
import pyglet
import os
os.environ['PYGLET_HEADLESS'] = 'False'
pyglet.options['headless'] = True

import register_envs

import shutil
import zipfile
import datetime

import subprocess

import gc
import pathlib
import torch

import ruamel.yaml as yaml

from pathlib import Path
from types import SimpleNamespace

import tools

import gymnasium as gym
import numpy as np

import ray

# Define env creator for Ray
from ray.tune import register_env
from ray.rllib.models import ModelCatalog

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

ray.init()

from ray.rllib.models import ModelCatalog
from ray_ppo import VisionNetwork  # Ensure correct relative import

ModelCatalog.register_custom_model("split_vision_model", VisionNetwork)

def get_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        return commit + ("-dirty" if dirty else "")
    except Exception as e:
        return f"Git info unavailable: {e}"
    
def make_config(path="ppo_configs.yaml"):
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    def dict_to_namespace(d):
        return SimpleNamespace(**{k: v for k, v in d.items()})

    yaml_parser = yaml.YAML(typ='safe', pure=True)
    config_yaml = yaml_parser.load((pathlib.Path(path)).read_text())

    config = {}
    recursive_update(config, config_yaml.get('defaults', {}))  # safe fallback
    config["ppo_config"] = config_yaml.get("ppo_config", {})

    return dict_to_namespace(config)

if torch.cuda.is_available():
    device = torch.device(0)
    print('CUDA initialised')
else:
    print("WARNING - CUDA not available - no hardware acceleration being used")

config = make_config()
tools.set_seed_everywhere(config.seed)

experiment_name = config.exp_name
exp_date = config.exp_date
path_root = config.path_root

results_path = f"{path_root}\\{experiment_name}\\results"
save_path = f"{path_root}\\{experiment_name}\\saved_models"
logdir = Path(f"{path_root}\\{experiment_name}\\logs")
data_path = f"{path_root}\\{experiment_name}\\dataset"

for p in [results_path, f"{results_path}\\{exp_date}", save_path, logdir, data_path]:
    os.makedirs(p, exist_ok=True)

# Create experiment folder
root_dir = Path(os.getcwd())
experiments_dir = root_dir / "experiments"
experiment_folder = experiments_dir / config.exp_date

# Delete if exists, then recreate
if experiment_folder.exists():
    shutil.rmtree(experiment_folder)
experiment_folder.mkdir(parents=True, exist_ok=True)

# ============================================================================
#   GLOBALS
# ----------------------------------------------------------------------------
logger = tools.Logger(logdir/exp_date, 0)

class NormalizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape,  # HWC
            dtype=np.float32,
        )

    def observation(self, obs):
        obs = obs.astype(np.float32) / 255.0  # Normalize to [0,1]
        return obs
    
class RewardScaler(gym.RewardWrapper):
    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale

# Load PPO config
ppo_config = config.ppo_config  # already unpacked as dict

def env_creator(env_config):
    env = gym.make(config.env_name, **config.maze_config)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NormalizeObsWrapper(env)
    env = RewardScaler(env, scale=1.0)
    return env

register_env("custom_env", env_creator)

ppo_config["env"] = "custom_env"

base_config = PPOConfig()
base_config = base_config.update_from_dict(ppo_config)
base_config = base_config.environment(env="custom_env")
base_config = base_config.framework("torch")

trainer = base_config.build()

torch.cuda.empty_cache()
gc.collect()

def log_train_metrics(result, logger, env_steps_so_far):
    """
    Log each training episode's reward and length individually to match Dreamer-style training logs.
    Args:
        result (dict): Result dict returned by trainer.train()
        logger (tools.Logger): Your existing custom logger
        env_steps_so_far (int): Total environment steps collected so far
    """
    stats = result.get("hist_stats", {})

    rewards = stats.get("episode_reward", [])[-ppo_config['num_envs_per_worker']:]
    lengths = stats.get("episode_lengths", [])[-ppo_config['num_envs_per_worker']:]

    if not rewards:
        return

    for reward, length in zip(rewards, lengths):
        logger.scalar("train_return", reward)
        logger.scalar("train_length", length)
        logger.write(step=env_steps_so_far)  # Use real env steps

# ============================================================================
#   REPRODUCABILITY
# ----------------------------------------------------------------------------

# Archive current configs.yaml + all .py files into a zip
archive_path = experiment_folder / f"{config.exp_date}_code_snapshot.zip"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    # Add configs.yaml
    config_path = root_dir / "configs.yaml"
    zipf.write(config_path, arcname="configs.yaml")
    
    # Add all .py files from the project root
    for py_file in root_dir.glob("*.py"):
        zipf.write(py_file, arcname=py_file.name)
    # Add all .ipynb files from the project root
    for ipynb_file in root_dir.glob("*.ipynb"):
        zipf.write(ipynb_file, arcname=ipynb_file.name)
# Get current datetime
now = datetime.datetime.now()
date_str = now.strftime("%d-%m-%Y")
time_str = now.strftime("%H:%M:%S")

# Write README
with open(experiment_folder / "README.txt", "w") as f:
    f.write(f"Experiment: {experiment_name}\n")
    f.write(f"Name: {exp_date}\n")
    f.write(f"Date: {date_str}\n")
    f.write(f"Start Time: {time_str}\n")
    f.write(f"Git: {get_git_commit()}\n")
    f.write(f"Notes:")

print("Git commit:", get_git_commit())

# ============================================================================
#   MAIN
# ----------------------------------------------------------------------------

def main():
    total_steps = 0
    target_steps = int(config.ppo_config['ppo_steps'])

    while total_steps < target_steps:
        result = trainer.train()
        steps_this_iter = result.get("timesteps_this_iter", result["timesteps_total"] - total_steps)
        total_steps = result["timesteps_total"]

        print(f"Total steps: {total_steps} / {target_steps}")
        log_train_metrics(result, logger, env_steps_so_far=total_steps)

if __name__ == "__main__":
    main()
