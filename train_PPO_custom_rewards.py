import gymnasium as gym
from stable_baselines3 import PPO
import reward_1, reward_2, reward_3

#env1 = gym.make("LunarLander-v2", render_mode = "rgb_array")
#env2 = gym.make("LunarLander-v2", render_mode = "rgb_array")
env3 = gym.make("LunarLander-v2", render_mode = "rgb_array")

#env1 = reward_1.CustomLunarLander(env1)
#env2 = reward_2.CustomLunarLander(env2)
env3 = reward_3.CustomLunarLander(env3)

#env1.reset()
#env2.reset()
env3.reset()

#model1 = PPO("MlpPolicy", env1, verbose=1, tensorboard_log="logs")
#model2 = PPO("MlpPolicy", env2, verbose=1, tensorboard_log="logs")
model3 = PPO("MlpPolicy", env3, verbose=1, tensorboard_log="logs")

TIME_STEPS = 10000
iters = 0

while TIME_STEPS * iters < 10000000:
    iters += 1
    #model1.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False, tb_log_name="PPO_reward_1_10M")
    #model2.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False, tb_log_name="PPO_reward_2_10M")
    model3.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False, tb_log_name="PPO_reward_3_10M_2")
    #model1.save(f"models/PPO_with_custom_rewards_1_10M/{iters * TIME_STEPS}")
    #model2.save(f"models/PPO_with_custom_rewards_2_10M/{iters * TIME_STEPS}")
    model3.save(f"models/PPO_with_custom_rewards_3_10M_2/{iters * TIME_STEPS}")
