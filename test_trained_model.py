import gymnasium as gym
from stable_baselines3 import DQN, A2C, PPO

model = PPO.load("models/PPO_no_modification/3800000")
env = gym.make("LunarLander-v2", render_mode = "human")

observation, info = env.reset()

for _ in range(100000):
    env.render()
    action, _ = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    # Reset the sim everytime the lander makes contact with the surface of the moon
    if terminated or truncated:
        observation, info = env.reset()
env.close()
