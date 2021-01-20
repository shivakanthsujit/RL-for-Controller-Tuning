import os

import torch
from control_env import GymSystem
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize
from utils import SaveImageCallback, SaveOnBestTrainingRewardCallback

print("CUDA Available: ", torch.cuda.is_available())
torch.autograd.set_detect_anomaly(True)

tag_name = "PID_Control_PPO"
model_dir = "./models"
log_dir = os.path.join("./logs", tag_name)

checkpoint_callback = CheckpointCallback(10000, model_dir, tag_name)
save_callback = SaveOnBestTrainingRewardCallback(check_freq=50000, log_dir=log_dir, verbose=1)
eval_env = GymSystem(deterministic=True)
save_image_callback = SaveImageCallback(eval_env=eval_env, eval_freq=20000, log_dir=log_dir)
callback = CallbackList([checkpoint_callback, save_callback, save_image_callback])

env = GymSystem()
env = make_vec_env(lambda: env, n_envs=4, monitor_dir=log_dir)
try:
    env = VecNormalize.load(os.path.join(log_dir, "vec_normalize.pkl"), env)
except RuntimeError:
    env = VecNormalize(env)
env = VecCheckNan(env, raise_exception=True)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

tsteps = 1_000_000
model.learn(tsteps, reset_num_timesteps=False, callback=callback, tb_log_name=tag_name)
