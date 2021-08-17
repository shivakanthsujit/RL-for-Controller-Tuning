import os
import argparse
import zipfile
import PIL
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch
import stable_baselines3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize

from cstr_control_env import GymCSTR, lamda
from wrappers import ActionRepeat, EarlyStopping
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='CSTRPID')
parser.add_argument('--type', default='servo')
parser.add_argument('--algo', default='PPO')
parser.add_argument('--logdir', default='logs')
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

env_model = args.model
env_module = import_module('cstr_control_env')
env_class =  getattr(env_module, env_model)

torch.autograd.set_detect_anomaly(True)
# print("CUDA Available: ", torch.cuda.is_available())

early_stopping = True
# print("Using Early Stopping: ", early_stopping)

action_repeat = True
action_repeat_value = 2
# print("Using Action Repeat: ", action_repeat, action_repeat_value)

use_sde = False
# print("Using gSDE: ", use_sde)

algo = args.algo
# print("Algorithm: ", algo)
tag_name = os.path.join(f"{env_model}", f"{algo}_AR_{action_repeat}_{action_repeat_value}_lamda_{lamda}_use_sde_{use_sde}_ES_{early_stopping}")
print("Run Name: ", tag_name)
print(f"Testing {args.type} response")

model_dir = "./models"
log_dir = os.path.join(args.logdir, tag_name, str(args.seed))

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


test_logs_dir = os.path.join(log_dir, 'test_files', args.type)
os.makedirs(test_logs_dir, exist_ok=True)

run_config = {
    'servo': {'disturbance': False, 'deterministic': True},
    'regulatory': {'disturbance': False, 'deterministic': True, 'load_change': True},
    'noise': {'disturbance': True, 'deterministic': True},
}

test_env = GymCSTR(
    system=env_class,
    **run_config[args.type]
)

if action_repeat:
    test_env = ActionRepeat(test_env, action_repeat_value)

obs = test_env.reset()
obss = [obs]
norm_obss = [model.env.normalize_obs(obs)]
actions = []
done = False
tot_r = 0.0
while not done:
    action, _ = model.predict(model.env.normalize_obs(obs), deterministic=True)
    obs, reward, done, info = test_env.step(action)
    obss.append(obs)
    norm_obss.append(model.env.normalize_obs(obs))
    actions.append(action)
    tot_r += reward
print("Test Reward: ", tot_r)

eval_img = test_env.render("rgb_array")
im = Image.fromarray(eval_img)
path = os.path.join(test_logs_dir, 'perf.png')
im.save(path)

axis, axis_name = test_env.system.get_axis()
labels = test_env.system.state_names.copy()
obss = np.array(obss)
norm_obss = np.array(norm_obss)

plt.figure(figsize=(16, 4))
lineObj = plt.plot(axis[test_env.system.ksp :: action_repeat_value], actions)
plt.legend(lineObj, ["Kp", "taui", "taud"])
plt.ylabel("Value")
plt.xlabel(axis_name)
plt.xlim(axis[0], axis[-1])
plt.title("Agent Outputs")
plt.show()

plt.figure(figsize=(16, 9))
plt.subplot(2, 1, 1)
lineObj = plt.plot(axis[test_env.system.ksp :: action_repeat_value], obss[:-1])
plt.legend(lineObj, labels)
plt.ylabel("Value")
plt.xlabel(axis_name)
plt.xlim(axis[0], axis[-1])
plt.grid()
plt.title("Unnormalized States")

plt.subplot(2, 1, 2)
lineObj = plt.plot(axis[test_env.system.ksp :: action_repeat_value], norm_obss[:-1])
plt.legend(lineObj, labels)
plt.ylabel("Value")
plt.xlabel(axis_name)
plt.xlim(axis[0], axis[-1])
plt.grid()
plt.title("Normalized States")

m_path = os.path.join(test_logs_dir, f"final_model_{algo}")
model.save(m_path)
obj_path = os.path.join(test_logs_dir, f'env_obj_{algo}.pkl')
import pickle
with open(obj_path, 'wb') as f:
  pickle.dump(test_env.env.system, f)