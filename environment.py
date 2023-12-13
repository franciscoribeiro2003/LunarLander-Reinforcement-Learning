import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

env = gym.make('CarRacing-v2', render_mode = 'rgb_array')
env = DummyVecEnv([lambda: env])

log_path = os.path.join('Train', 'Logs')
model = PPO('CnnPolicy', env, verbose = 1, tensorboard_log = log_path)

TIME_STEPS = 1500000
iters = 0

while True:
    iters += 1
    model.learn(total_timesteps = TIME_STEPS, reset_num_timesteps = False, tb_log_name = 'PPO')
    model.save('Train/Models/CarRacing-v2_' + str(iters))
