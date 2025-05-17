from dino_wm.utils.get_model import get_model
from dino_wm.planning.objectives import create_objective_fn
from dino_wm.planning.mpc import MPCPlanner
from dino_wm.planning.cem import CEMPlanner
from einops import rearrange, repeat
import einops
from dino_wm.utils.utils import move_to_device
from dino_wm.env.venv import SubprocVectorEnv
import gymnasium as gym
import gym_pusht
import torch
import numpy as np
from dino_wm.models.visual_world_model import VWorldModel
from dino_wm.utils.preprocessor import Preprocessor

n_envs = 1
frameskip = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

wm, dataset, data_preprocessor = get_model("/home/ianchuang/dino_wm/outputs/checkpoints", "pusht", device)
wm : VWorldModel = wm.to(device)

# gym.vector.SyncVectorEnv
env = gym.vector.AsyncVectorEnv(
    [
        lambda: gym.make(
            "gym_pusht/PushT-v0", 
            disable_env_checker=True, 
            obs_type="visual_proprio", 
            render_mode="rgb_array",
            observation_width=224,
            observation_height=224,
        )
        for _ in range(n_envs)
    ]
)
obs, info = env.reset()


batch = {
    k: np.expand_dims(v, axis=1).astype(np.float32)
    for k, v in obs.items()
}
batch = data_preprocessor.transform_obs(batch)
batch = {
    k: v.to(device)
    for k, v in batch.items()
}

z_obs_g = wm.encode_obs(batch)




# env = SubprocVectorEnv(
#     [
#         lambda: gym.make(
#             "pusht", with_velocity=True, with_target=True
#         )
#         for _ in range(n_envs)
#     ]
# )
