import gymnasium as gym
from stable_baselines3                   import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util   import make_vec_env
import os

env = gym.make("LunarLander-v2", render_mode = "human")

# reset the environment to initial state
observation = env.reset()

for _ in range(10000):
  print("---------------------------")

  # take a random action
  action = env.action_space.sample()
  print("Action taken: ", action)

  observation, reward, done, truncated, info = env.step(action)
  print("Observation: ", observation)
  print("Reward: ", reward)

  # If the game is done (in our case we land, crashed or timeout)
  if done:
    # Reset the environment
    print("Environment is reset")
    observation = env.reset()
