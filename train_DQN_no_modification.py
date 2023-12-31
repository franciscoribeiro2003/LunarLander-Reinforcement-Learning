import gymnasium as gym
from stable_baselines3                   import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util   import make_vec_env
import os

env = gym.make("LunarLander-v2", render_mode="rgb_array")
env.reset()

model = DQN("MlpPolicy", env, verbose = 1, tensorboard_log = "logs")

TIME_STEPS = 10000

iters = 0

while TIME_STEPS * iters < 5000000:
    iters += 1
    model.learn(total_timesteps = TIME_STEPS, reset_num_timesteps = False, tb_log_name = "DQN_no_modification")
    model.save(f"models/DQN_no_modification/{iters * TIME_STEPS}")
