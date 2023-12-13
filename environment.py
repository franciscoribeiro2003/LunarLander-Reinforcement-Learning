import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os


env = gym.make('CarRacing-v2', render_mode = 'rgb_array')
env = DummyVecEnv([lambda: env])

log_path = os.path.join('Train', 'Logs')
model = PPO('CnnPolicy', env, verbose = 1, tensorboard_log = log_path)

model.learn(total_timesteps = 2000000)

ppo_path = os.path.join('Train', 'Models', 'PPO_2M')
model.save(ppo_path)
