import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import reward_1

env = gym.make("LunarLander-v2", render_mode = "rgb_array")

env = reward_1.CustomLunarLander(env)

env.reset()

env = DummyVecEnv([lambda: env])

model = DQN("MlpPolicy", env, verbose = 1, tensorboard_log = "logs")

TIME_STEPS = 10000
iters = 0

while TIME_STEPS * iters < 5000000:
    iters += 1
    model.learn(total_timesteps = TIME_STEPS, reset_num_timesteps = False, tb_log_name = "DQN_reward_1_5M")
    model.save(f"models/PPO_with_custom_rewards/{iters * TIME_STEPS}")
