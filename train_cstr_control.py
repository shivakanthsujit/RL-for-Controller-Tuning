import os

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize

from cstr_control_env import CSTRFuzzyPID2, CSTRFuzzyPID3, GymCSTR, lamda
from utils import SaveImageCallback, SaveOnBestTrainingRewardCallback
from wrappers import ActionRepeat, EarlyStopping

torch.autograd.set_detect_anomaly(True)
print("CUDA Available: ", torch.cuda.is_available())

early_stopping = True
print("Using Early Stopping: ", early_stopping)

action_repeat = True
action_repeat_value = 2
print("Using Action Repeat: ", action_repeat, action_repeat_value)

use_sde = False
print("Using gSDE: ", use_sde)

algo = "PPO"
print("Algorithm: ", algo)
tag_name = f"CSTRFuzzyPID2_{algo}_AR_{action_repeat}_{action_repeat_value}_lamda_{lamda}_use_sde_{use_sde}_ES_{early_stopping}"
print("Run Name: ", tag_name)

model_dir = "./models"
log_dir = os.path.join("./logs", tag_name)

checkpoint_callback = CheckpointCallback(100000, model_dir, tag_name)
save_callback = SaveOnBestTrainingRewardCallback(check_freq=20000, log_dir=log_dir, verbose=1)

# eval_env = GymCSTR(system=CSTRFuzzyPID)
eval_env = GymCSTR(system=CSTRFuzzyPID2)
if early_stopping:
    eval_env = EarlyStopping(eval_env)
if action_repeat:
    eval_env = ActionRepeat(eval_env, action_repeat_value)
save_image_callback = SaveImageCallback(eval_env=eval_env, eval_freq=50000, log_dir=log_dir)

callback = CallbackList([checkpoint_callback, save_callback, save_image_callback])
print(callback.callbacks)

# env = GymCSTR(system=CSTRFuzzyPID)
env = GymCSTR(system=CSTRFuzzyPID2)
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

if algo == "PPO":
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/", use_sde=use_sde)
elif algo == "SAC":
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/", use_sde=use_sde)
else:
    raise NotImplementedError()

best_model_path = os.path.join(log_dir, "best_model.zip")
if os.path.exists(best_model_path):
    print("Found previous checkpoint. Loading from checkpoint.")
    if algo == "PPO":
        model = PPO.load(best_model_path, env)
    elif algo == "SAC":
        model = SAC.load(best_model_path, env)
    else:
        raise NotImplementedError()
print(model)

tsteps = 2_000_000
model.learn(tsteps, reset_num_timesteps=False, callback=callback, tb_log_name=tag_name)
