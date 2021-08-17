import os
import argparse

import torch
import stable_baselines3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize

from cstr_control_env import GymCSTR, lamda
from utils import SaveImageCallback, SaveOnBestTrainingRewardCallback, set_seed
from wrappers import ActionRepeat, EarlyStopping
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='CSTRPID')
parser.add_argument('--algo', default='PPO')
parser.add_argument('--steps', default=500000, type=int)
parser.add_argument('--logdir', default='logs')
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

set_seed(args.seed, torch.cuda.is_available())

env_model = args.model
env_module = import_module('cstr_control_env')
env_class =  getattr(env_module, env_model)

torch.autograd.set_detect_anomaly(True)
print("CUDA Available: ", torch.cuda.is_available())

early_stopping = True
print("Using Early Stopping: ", early_stopping)

action_repeat = True
action_repeat_value = 2
print("Using Action Repeat: ", action_repeat, action_repeat_value)

use_sde = False
print("Using gSDE: ", use_sde)

algo = args.algo
print("Algorithm: ", algo)
tag_name = os.path.join(f"{env_model}", f"{algo}_AR_{action_repeat}_{action_repeat_value}_lamda_{lamda}_use_sde_{use_sde}_ES_{early_stopping}")
print("Run Name: ", tag_name)

model_dir = "./models"
log_dir = os.path.join(args.logdir, tag_name, str(args.seed))

checkpoint_callback = CheckpointCallback(50000, model_dir, tag_name)
save_callback = SaveOnBestTrainingRewardCallback(check_freq=20000, log_dir=log_dir, verbose=1)

eval_env = GymCSTR(system=env_class)
if early_stopping:
    eval_env = EarlyStopping(eval_env)
if action_repeat:
    eval_env = ActionRepeat(eval_env, action_repeat_value)
save_image_callback = SaveImageCallback(eval_env=eval_env, eval_freq=50000, log_dir=log_dir)

callback = CallbackList([checkpoint_callback, save_callback, save_image_callback])
print(callback.callbacks)

env = GymCSTR(system=env_class)
if early_stopping:
    env = EarlyStopping(env)
if action_repeat:
    env = ActionRepeat(env, action_repeat_value)
env = make_vec_env(lambda: env, n_envs=1, monitor_dir=log_dir)

if os.path.exists(os.path.join(log_dir, "vec_normalize.pkl")):
    print("Found VecNormalize Stats. Using stats")
    env = VecNormalize.load(os.path.join(log_dir, "vec_normalize.pkl"), env)
else:
    print("No previous stats found. Using new VecNormalize instance.")
    env = VecNormalize(env)
env = VecCheckNan(env, raise_exception=True)

algo_class =  getattr(stable_baselines3, algo)
model = algo_class("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/", seed=args.seed)

best_model_path = os.path.join(log_dir, "best_model.zip")
if os.path.exists(best_model_path):
    print("Found previous checkpoint. Loading from checkpoint.")
    model = algo_class.load(best_model_path, env)

print(model)

tsteps = args.steps
model.learn(tsteps, reset_num_timesteps=False, callback=callback, tb_log_name=tag_name)
